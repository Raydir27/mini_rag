# src/mini_rag/app.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Tuple

from .indexer import MiniIndexer


def load_documents_from_txt_folder(folder: str) -> Tuple[list, list]:
    p = Path(folder)
    docs = []
    ids = []
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {folder}")
    for fname in sorted(p.iterdir()):
        if fname.suffix.lower() != ".txt":
            continue
        with fname.open("r", encoding="utf-8") as f:
            docs.append(f.read().strip())
            ids.append(fname.name)
    return ids, docs


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mini_rag")
    parser.add_argument("--build-from", help="folder with .txt files to build index from")
    parser.add_argument("--index-dir", default="indexdir", help="where to save/load index (directory)")
    parser.add_argument("--query", help="query to search the index")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    args = parser.parse_args(args=argv)

    idx_dir = Path(args.index_dir)
    indexer = MiniIndexer(model_name=args.model)

    if args.build_from:
        ids, docs = load_documents_from_txt_folder(args.build_from)
        if not docs:
            print("No .txt documents found in folder. Exiting.", file=sys.stderr)
            return 1
        print(f"Building index from {len(docs)} docs using model {args.model}...")
        indexer.build(docs, ids)
        indexer.save(idx_dir)
        print(f"Index built and saved to {idx_dir}")
    else:
        # load existing index
        try:
            indexer.load(idx_dir)
            print(f"Loaded index from {idx_dir} ({len(indexer.docs)} docs).")
        except Exception as e:
            print(f"Failed to load index from {idx_dir}: {e}", file=sys.stderr)
            return 2

    if args.query:
        print(f"Searching for: {args.query!r} (top_k={args.topk})")
        results = indexer.search(args.query, top_k=args.topk)
        if not results:
            print("No results.")
            return 0
        for rank, (doc_id, doc, score) in enumerate(results, start=1):
            print(f"\n{rank}. id={doc_id} score={score:.4f}\n---\n{doc[:400]}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
