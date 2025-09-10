# src/indexer.py
import argparse
import faiss
import numpy as np
import os
from src.docs_loader import load_pdf_text, chunk_text
from src.embed import Embedder

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def main(pdf_path: str, index_out: str, emb_out: str):
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    embedder = Embedder()
    embeddings = embedder.embed(chunks)  # shape (n, d)

    # save embeddings and text (we'll save texts as .npy of object dtype)
    np.save(emb_out, embeddings)
    # Save texts too
    np.save(emb_out.replace(".npy", "_texts.npy"), np.array(chunks, dtype=object))

    # build index
    index = build_index(embeddings)
    faiss.write_index(index, index_out)
    print(f"Saved index -> {index_out}, embeddings -> {emb_out}, texts -> {emb_out.replace('.npy','_texts.npy')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--index-out", default="index.faiss")
    parser.add_argument("--emb-out", default="embeddings.npy")
    args = parser.parse_args()
    main(args.pdf_path, args.index_out, args.emb_out)
