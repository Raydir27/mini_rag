# indexer.py

from .embedder import get_dim
import json
import numpy as np
from pathlib import Path
from typing import List

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

INDEX_DIR = Path('index')
INDEX_DIR.mkdir(exist_ok=True)
dim = get_dim()  # get from embedder



def build_faiss_index(embeddings: List[List[float]], metadatas: List[dict], dim: int, index_path: str = 'index/faiss.index'):
    xb = np.array(embeddings).astype('float32')
    if xb.shape[1] != dim:
        raise ValueError(f"Dimension mismatch: embeddings have dimension {xb.shape[1]}, but 'dim' is {dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(xb)
    faiss.write_index(index, index_path)
    # persist metadata
    with open('index/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    print('Index and metadata saved.')




def load_index(index_path: str = 'index/faiss.index'):
    index = faiss.read_index(index_path)
    with open('index/metadata.json', 'r', encoding='utf-8') as f:
        metas = json.load(f)
    return index, metas




def query_index(index, metas, query_embedding, top_k=5):
    q = np.array([query_embedding]).astype('float32')
    results = []
    if _HAS_FAISS and index is not None:
        D, I = index.search(q, top_k)
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = metas[idx]
            results.append({'score': float(score), 'metadata': m})
        return results
    # fallback: numpy cosine on normalized embeddings stored in metas
    embeddings = np.stack([m['embedding'] for m in metas], axis=0)  # ensure metadata contains embeddings
    # normalize if not already
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embeddings = embeddings / norms
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    sims = embeddings @ qn.T
    top_idx = np.argsort(-sims.squeeze())[:top_k]
    for i in top_idx:
        results.append({'score': float(sims[i,0]), 'metadata': metas[i]})
    return results