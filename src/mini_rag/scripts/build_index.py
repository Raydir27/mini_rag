#!/usr/bin/env python3
"""
scripts/build_index.py

Usage (from repo root):
# as a module (recommended):
python -m src.mini_rag.scripts.build_index --data ./data --index-dir ./index

# or directly (if running from src/):
python src\mini_rag\scripts\build_index.py --data .\data --index-dir .\index
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Flexible import helper: try several import paths so the script works both as:
# - a module: python -m src.mini_rag.scripts.build_index
# - a direct script: python src\mini_rag\scripts\build_index.py
def try_import():
    """
    Returns a dict of imported functions/modules:
      - prepare_documents (from ingest)
      - embed_texts_numpy or embed_texts (from embedder)
      - get_dim (optional)
      - build_faiss_index (from indexer)
    """
    imported = {}
    tried = []
    # candidate module paths to try
    candidates = [
        ("src.mini_rag.ingest", "prepare_documents"),
        ("mini_rag.ingest", "prepare_documents"),
        ("ingest", "prepare_documents"),
    ]
    for mod_path, sym in candidates:
        try:
            mod = __import__(mod_path, fromlist=[sym])
            imported['prepare_documents'] = getattr(mod, sym)
            break
        except Exception as e:
            tried.append((mod_path, e))
    # embedder
    embed_candidates = [
        ("src.mini_rag.embedder", None),
        ("mini_rag.embedder", None),
        ("embedder", None),
    ]
    for mod_path, _ in embed_candidates:
        try:
            mod = __import__(mod_path, fromlist=['embed_texts_numpy', 'embed_texts', 'get_dim'])
            # prefer numpy variant if available
            imported['embed_texts_numpy'] = getattr(mod, 'embed_texts_numpy', None)
            imported['embed_texts'] = getattr(mod, 'embed_texts', None)
            imported['get_dim'] = getattr(mod, 'get_dim', None)
            break
        except Exception as e:
            tried.append((mod_path, e))
    # indexer
    index_candidates = [
        ("src.mini_rag.indexer", "build_faiss_index"),
        ("mini_rag.indexer", "build_faiss_index"),
        ("indexer", "build_faiss_index"),
    ]
    for mod_path, sym in index_candidates:
        try:
            mod = __import__(mod_path, fromlist=[sym])
            imported['build_faiss_index'] = getattr(mod, sym)
            # optional: also accept save/load helpers
            imported['save_metadata'] = getattr(mod, 'save_metadata', None)
            break
        except Exception as e:
            tried.append((mod_path, e))
    # last resort: try adding repo root to sys.path and import using package name
    if 'prepare_documents' not in imported or 'build_faiss_index' not in imported:
        repo_root = Path(__file__).resolve().parents[3]  # adjust to repo root from scripts/ file
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        # Try once more using src.mini_rag.*
        try:
            import importlib
            mod = importlib.import_module("src.mini_rag.ingest")
            imported['prepare_documents'] = getattr(mod, 'prepare_documents')
            mod = importlib.import_module("src.mini_rag.embedder")
            imported['embed_texts_numpy'] = getattr(mod, 'embed_texts_numpy', None)
            imported['embed_texts'] = getattr(mod, 'embed_texts', None)
            imported['get_dim'] = getattr(mod, 'get_dim', None)
            mod = importlib.import_module("src.mini_rag.indexer")
            imported['build_faiss_index'] = getattr(mod, 'build_faiss_index')
        except Exception as e:
            # we'll surface errors later
            tried.append(("src.mini_rag.* attempt", e))

    imported['_tried_imports'] = tried
    return imported

def save_embeddings_npy(embs, path: Path):
    import numpy as np
    arr = None
    if isinstance(embs, list):
        arr = np.array(embs, dtype="float32")
    else:
        arr = embs.astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)
    print(f"Saved embeddings to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", required=True, help="Path to documents root (passed to ingest.prepare_documents)")
    parser.add_argument("--index-dir", "-i", default="./index", help="Directory to write faiss index and metadata")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--save-embeddings", action="store_true", help="Also save embeddings as .npy in index dir")
    parser.add_argument("--topk-sanity", type=int, default=3, help="Run a quick sanity query (optional)")
    args = parser.parse_args()

    imported = try_import()
    if 'prepare_documents' not in imported or 'build_faiss_index' not in imported:
        print("ERROR: Failed to import project modules. Import attempts:")
        for item in imported.get('_tried_imports', []):
            print("  ", item)
        print("\nMake sure you're running the script from the repo root or using the -m module form:")
        print("  python -m src.mini_rag.scripts.build_index --data ./data --index-dir ./index")
        sys.exit(2)

    prepare_documents = imported['prepare_documents']
    build_faiss_index = imported['build_faiss_index']
    embed_numpy = imported.get('embed_texts_numpy')
    embed_list = imported.get('embed_texts')
    get_dim = imported.get('get_dim')

    data_root = Path(args.data)
    if not data_root.exists():
        print(f"ERROR: data path {data_root} does not exist.")
        sys.exit(2)

    print("Preparing documents (chunking)...")
    docs = prepare_documents(str(data_root))
    texts = [d['text'] for d in docs]
    print(f"Prepared {len(texts)} chunks from {args.data}")

    print("Computing embeddings...")
    if embed_numpy is not None:
        embs = embed_numpy(texts, batch_size=args.batch_size)
        # embs is numpy array (N x dim)
        # convert to list-of-lists for compatibility with simple indexer
        try:
            emb_list = embs.tolist()
        except Exception:
            import numpy as np
            emb_list = np.array(embs).astype("float32").tolist()
    elif embed_list is not None:
        emb_list = embed_list(texts, batch_size=args.batch_size)
        embs = None
    else:
        print("ERROR: No embedder found (neither embed_texts_numpy nor embed_texts).")
        sys.exit(2)

    # metadata
    metas = []
    for i, d in enumerate(docs):
        m = {
            "id": d.get("id") or f"doc_{i}",
            "source": d.get("source"),
            "chunk_index": d.get("chunk_index"),
            # DO NOT store embeddings here in large projects (kept small here for clarity)
        }
        metas.append(m)

    # ensure index dir exists
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # determine embedding dim
    if get_dim is not None:
        dim = get_dim()
    else:
        # fallback: infer from first embedding
        dim = len(emb_list[0]) if len(emb_list) > 0 else None
    if dim is None:
        print("ERROR: could not determine embedding dimension.")
        sys.exit(2)

    print(f"Building FAISS index (dim={dim}) ...")
    index_path = str(index_dir / "faiss.index")
    # call build_faiss_index(embeddings, metadatas, dim, index_path)
    build_faiss_index(emb_list, metas, dim, index_path=index_path)

    # save metadata
    meta_path = index_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {meta_path}")

    # optional: save embeddings as .npy
    if args.save_embeddings:
        emb_path = index_dir / "embeddings.npy"
        save_embeddings_npy(embs if embed_numpy is not None else emb_list, emb_path)

    print("Index build complete.")
    print(f"Index: {index_path}")
    print(f"Metadata: {meta_path}")
    if args.save_embeddings:
        print(f"Embeddings: {emb_path}")

if __name__ == "__main__":
    main()

###
# Notes about the script

# It's written defensively: tries multiple import paths so it works when run as python -m src.mini_rag.scripts.build_index or as a direct script.

# It calls your repo's prepare_documents (ingest), embed functions (prefer embed_texts_numpy if available), and build_faiss_index.

# Writes index/faiss.index and index/metadata.json (path configurable via --index-dir).

# Optional --save-embeddings will write index/embeddings.npy.
###