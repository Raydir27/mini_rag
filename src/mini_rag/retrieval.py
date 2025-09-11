# File: src/mini_rag/retrieval.py
"""
Retrieval utilities for mini_rag (Day 3)

This module provides a flexible loader/searcher for a vector index built on Day 2.
It supports Chroma, FAISS, and a numpy fallback. Embeddings are produced using
SentenceTransformers (recommended) by default; a small fallback embedding is
kept for extreme edge-cases.

Functions
---------
- load_index(path_or_config): returns an index object (opaque) and an `embed_fn` callable
- search(index_obj, embed_fn, query, top_k=5): returns list of results with scores and metadata

"""
from typing import Callable, Dict, Any, List, Tuple
import os
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Optional backends
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# SentenceTransformers for embeddings (preferred)
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

# Load ST model once (configurable via env var)
MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
ST_MODEL = None
if ST_AVAILABLE:
    try:
        ST_MODEL = SentenceTransformer(MODEL_NAME)
        logger.info("Loaded SentenceTransformer model: %s", MODEL_NAME)
    except Exception as e:
        logger.warning("Failed to load SentenceTransformer (%s): %s", MODEL_NAME, e)
        ST_MODEL = None
else:
    logger.warning("sentence-transformers not installed; embeddings fallback will be used")


def st_embed_fn(text: str) -> np.ndarray:
    """Embed text using the loaded SentenceTransformer model.

    Returns a 1-D float32 numpy array.
    """
    if ST_MODEL is None:
        raise RuntimeError("SentenceTransformer model not loaded")
    emb = ST_MODEL.encode([text], convert_to_numpy=True)[0]
    return emb.astype('float32')


def default_embed_fn(text: str) -> np.ndarray:
    """A deterministic small fallback embedding used only if ST is unavailable.

    NOT for production use. Returns a fixed-size float32 vector.
    """
    vec = np.frombuffer(text.encode('utf-8')[:512].ljust(512, b'\0'), dtype=np.uint8).astype('float32')
    vec = vec.mean().reshape(1) * np.ones(32, dtype='float32')
    return vec


# ---------------------- Loading helpers ----------------------

def load_index(path: str) -> Tuple[Any, Callable[[str], np.ndarray]]:
    """Load a vector index from a path and return (index_obj, embed_fn).

    The function attempts the following (in order):
    1. If 'chroma' folder exists and chromadb is available, open a Chroma client there.
    2. If a FAISS index file exists and faiss is available, load it together with a metadata json.
    3. If a numpy .npz exists with 'embeddings' and 'docs', load that as a simple index.

    If a SentenceTransformer model is available it will be returned as the embed_fn; otherwise
    a small fallback embedding function is returned.
    """
    embed_fn = st_embed_fn if ST_MODEL is not None else default_embed_fn

    # Chroma option
    if os.path.isdir(path) and CHROMA_AVAILABLE:
        try:
            client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=path))
            colls = client.list_collections()
            coll = None
            if colls:
                # pick the first collection
                coll = client.get_collection(colls[0].name)
            logger.info("Loaded Chroma index from %s", path)
            return ({'type': 'chroma', 'client': client, 'collection': coll}, embed_fn)
        except Exception as e:
            logger.warning("Chroma present but failed to load: %s", e)

    # FAISS option
    if any(path.endswith(ext) for ext in ('.index', '.faiss')) and FAISS_AVAILABLE:
        try:
            idx = faiss.read_index(path)
            meta_path = os.path.splitext(path)[0] + '.meta.json'
            docs = []
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    docs = json.load(f)
            logger.info("Loaded FAISS index from %s", path)
            return ({'type': 'faiss', 'index': idx, 'docs': docs}, embed_fn)
        except Exception as e:
            logger.warning("FAISS present but failed to load: %s", e)

    # Numpy fallback
    npz_path = path if path.endswith('.npz') else os.path.join(path, 'index.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        embeddings = data['embeddings']
        docs = data['docs'].tolist()
        logger.info("Loaded numpy index from %s", npz_path)
        return ({'type': 'numpy', 'embeddings': embeddings.astype('float32'), 'docs': docs}, embed_fn)

    raise FileNotFoundError(f"No index found at {path} (looked for chroma folder, .index/.faiss, or .npz)")


# ---------------------- Search functions ----------------------

def search(index_obj: Dict[str, Any], embed_fn: Callable[[str], np.ndarray], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search the provided index object using the provided embed_fn.

    Returns a list of results: each is a dict with keys: 'id' (optional), 'score', 'doc' (text), and 'meta' (optional)
    """
    qvec = embed_fn(query).astype('float32')

    t = index_obj.get('type')
    if t == 'chroma':
        coll = index_obj['collection']
        if coll is None:
            return []
        results = coll.query(query_texts=[query], n_results=top_k)
        out = []
        for i in range(len(results.get('ids', [[]])[0])):
            out.append({'id': results['ids'][0][i], 'score': results['distances'][0][i], 'doc': results['documents'][0][i], 'meta': results['metadatas'][0][i] if 'metadatas' in results else None})
        return out

    if t == 'faiss':
        idx = index_obj['index']
        docs = index_obj.get('docs', [])
        D, I = idx.search(qvec.reshape(1, -1), top_k)
        out = []
        for score, i in zip(D[0], I[0]):
            doc = docs[i] if i < len(docs) else None
            out.append({'id': int(i), 'score': float(score), 'doc': doc, 'meta': None})
        return out

    if t == 'numpy':
        embeddings = index_obj['embeddings']
        docs = index_obj['docs']
        q = qvec.reshape(-1)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        def norm(a):
            n = np.linalg.norm(a, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return a / n
        qn = q / (np.linalg.norm(q) + 1e-12)
        em_n = norm(embeddings)
        sims = (em_n @ qn).astype('float32')
        ids = np.argsort(-sims)[:top_k]
        out = []
        for i in ids:
            out.append({'id': int(i), 'score': float(sims[i]), 'doc': docs[i], 'meta': None})
        return out

    raise ValueError(f"Unsupported index type: {t}")


# ---------------------- Utility: pretty print ----------------------

def format_results(results: List[Dict[str, Any]]) -> str:
    """Return a human-friendly string representation of search results."""
    lines = []
    for r in results:
        lines.append(f"- score={r.get('score'):.4f} id={r.get('id')}\n  {r.get('doc')[:400]}\n")
    return "\n".join(lines)


# ---------------------- Simple CLI for quick testing ----------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', '-i', required=True, help='Path to index folder/file (.npz, chroma dir, .index)')
    parser.add_argument('--query', '-q', required=True)
    parser.add_argument('--k', '-k', type=int, default=5)
    args = parser.parse_args()

    idx, embed = load_index(args.index)
    res = search(idx, embed, args.query, top_k=args.k)
    print(format_results(res))
