# src/query_cli.py
import argparse
import faiss
import numpy as np
from mini_rag.embed import Embedder

def load_texts(emb_path):
    return np.load(emb_path.replace(".npy", "_texts.npy"), allow_pickle=True)

def interactive_query(index_path: str, emb_path: str, topk: int = 3):
    index = faiss.read_index(index_path)
    query_text = input("Enter your query: ").strip()
    embedder = Embedder()
    q_emb = embedder.embed([query_text])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, topk)
    texts = load_texts(emb_path)
    for rank, idx in enumerate(I[0]):
        score = float(D[0][rank])
        snippet = texts[idx]
        print(f"\nRank {rank+1} (score {score:.4f}):\n{snippet}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--emb-path", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    interactive_query(args.index_path, args.emb_path, args.topk)
