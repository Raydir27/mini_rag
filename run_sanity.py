# File: run_sanity.py
"""
A small standalone sanity check script for embeddings and retrieval.

- Loads a SentenceTransformer model (defaults to all-MiniLM-L6-v2).
- Embeds a few sample docs and a query.
- Computes cosine similarities and prints them sorted.

Run:
    python run_sanity.py
Optionally set env var SENTENCE_TRANSFORMER_MODEL to override the model name.
"""
import os
import numpy as np

# fallback embedding
def fallback_embed(text: str) -> np.ndarray:
    vec = np.frombuffer(text.encode('utf-8')[:512].ljust(512, b'\0'), dtype=np.uint8).astype('float32')
    vec = vec.mean().reshape(1) * np.ones(32, dtype='float32')
    return vec

# try to load SentenceTransformer
ST_MODEL = None
MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
try:
    from sentence_transformers import SentenceTransformer
    ST_MODEL = SentenceTransformer(MODEL_NAME)
    print(f"[OK] Loaded SentenceTransformer model: {MODEL_NAME}")
except Exception as e:
    print(f"[WARN] Could not load SentenceTransformer ({MODEL_NAME}): {e}")
    print("[INFO] Using fallback embedding instead.")


def embed_fn(text: str) -> np.ndarray:
    if ST_MODEL is not None:
        return ST_MODEL.encode([text], convert_to_numpy=True)[0].astype('float32')
    return fallback_embed(text)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = a / (np.linalg.norm(a) + 1e-12)
    nb = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(na, nb))


def main():
    docs = [
        "Sentence transformers convert sentences into embeddings for semantic search.",
        "This document explains how to build a vector index and query it.",
        "Cooking recipes and ingredients have nothing to do with sentence transformers."
    ]
    query = "how to search with sentence embeddings"

    emb_docs = [embed_fn(d) for d in docs]
    emb_query = embed_fn(query)

    sims = [cosine_sim(emb_query, e) for e in emb_docs]

    results = sorted(zip(range(len(docs)), docs, sims), key=lambda x: -x[2])

    print("\n[Sanity Check Results]\n")
    print(f"Query: {query}\n")
    for i, doc, score in results:
        print(f"Doc {i} | score={score:.4f}\n{doc}\n")


if __name__ == "__main__":
    main()
