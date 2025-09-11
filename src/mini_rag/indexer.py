# src/mini_rag/indexer.py
from __future__ import annotations
from typing import List, Optional, Tuple
from pathlib import Path
import os
import numpy as np

# Try optional faiss
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

from sentence_transformers import SentenceTransformer


class MiniIndexer:
    """
    Lightweight indexer that supports:
      - SentenceTransformer embeddings
      - Optional FAISS IndexFlatIP (if faiss available)
      - Numpy-based fallback search (cosine via normalized dot)
      - Save / load of metadata + embeddings
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # shape (n, d), float32
        self.index = None  # faiss index or None

    def embed_docs(self, docs: List[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents (returns float32 numpy array)."""
        embs = self.model.encode(docs, show_progress_bar=show_progress, convert_to_numpy=True)
        # ensure float32
        return np.asarray(embs, dtype="float32")

    def build(self, docs: List[str], ids: Optional[List[str]] = None, normalize: bool = True):
        """Build index from docs. If faiss present, builds IndexFlatIP on normalized vectors."""
        if ids is None:
            ids = [str(i) for i in range(len(docs))]
        assert len(ids) == len(docs), "ids and docs must be same length"
        self.ids = ids
        self.docs = docs
        self.embeddings = self.embed_docs(docs)
        if normalize:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = (self.embeddings / norms).astype("float32")

        d = int(self.embeddings.shape[1])
        if _HAS_FAISS:
            # FAISS IndexFlatIP expects float32
            self.index = faiss.IndexFlatIP(d)
            # faiss assumes vectors are not necessarily normalized; we already normalized above
            self.index.add(self.embeddings)
        else:
            self.index = None

    def save(self, path: str | Path):
        """Save ids, docs, embeddings and faiss index (if available) into a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "ids.npy", np.array(self.ids))
        # save docs as object array to preserve strings
        np.save(path / "docs.npy", np.array(self.docs, dtype=object))
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save. Call build() first.")
        np.save(path / "embeddings.npy", self.embeddings)
        if _HAS_FAISS and self.index is not None:
            try:
                faiss.write_index(self.index, str(path / "faiss.index"))
            except Exception:
                # fail gracefully — index still saved by embeddings.npy
                pass

    def load(self, path: str | Path):
        """Load ids, docs, embeddings and faiss index (if present)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index directory not found: {path}")
        self.ids = list(np.load(path / "ids.npy", allow_pickle=True))
        self.docs = list(np.load(path / "docs.npy", allow_pickle=True))
        self.embeddings = np.load(path / "embeddings.npy", allow_pickle=True).astype("float32")
        if _HAS_FAISS:
            try:
                self.index = faiss.read_index(str(path / "faiss.index"))
            except Exception:
                self.index = None
        else:
            self.index = None

    def _query_faiss(self, q_emb: np.ndarray, top_k: int) -> List[Tuple[str, str, float]]:
        D, I = self.index.search(q_emb.astype("float32"), top_k)
        results: List[Tuple[str, str, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((self.ids[int(idx)], self.docs[int(idx)], float(score)))
        return results

    def _query_numpy(self, q_emb: np.ndarray, top_k: int) -> List[Tuple[str, str, float]]:
        # ensure embeddings present
        if self.embeddings is None:
            return []
        # q_emb expected normalized already
        sims = (self.embeddings @ q_emb.T).squeeze()  # shape (n,)
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.ids[int(i)], self.docs[int(i)], float(sims[int(i)])) for i in top_idx]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search query and return list of tuples (id, doc, score). Scores are cosine-like if embeddings normalized."""
        if self.embeddings is None:
            raise RuntimeError("Index not built/loaded. Call build() or load().")
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        # normalize q
        qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        if _HAS_FAISS and self.index is not None:
            return self._query_faiss(qn, top_k)
        return self._query_numpy(qn, top_k)
