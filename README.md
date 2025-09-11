# mini_rag — small local RAG demo

A compact Retrieval-Augmented-Generation (RAG) app for local Q&A over PDFs / text files.  
User uploads PDFs / text files → text is extracted & split → embeddings are created → a vector index is built and queried via CLI or FastAPI.

[![CI_dev](https://github.com/Raydir27/mini_rag/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/ci.yml)

---

## Features
- **Document loading**
  - PDF parsing via [`pypdf`](https://pypi.org/project/pypdf/).
  - Safe text chunking with overlap (memory-safe implementation).
- **Embeddings**
  - Local embeddings with `sentence-transformers` (`all-MiniLM-L6-v2` by default).
  - OpenAI embeddings supported if API key is provided.
- **Vector search**
  - FAISS-based index (with NumPy fallback on Windows).
- **Interfaces**
  - CLI entrypoints (`query_cli.py`, `python -m mini_rag.app`).
  - FastAPI server for indexing and querying.
- **Testing**
  - Pytest-based test suite (`src/mini_rag/tests`).
  - Parallel execution with `pytest-xdist`.
  - Unit tests for loaders, chunking, indexing, and the pipeline.
- **CI/CD**
  - GitHub Actions workflow (`.github/workflows/ci.yml`) runs tests on push/PR.
  - Split requirements:
    - `requirements.txt` — runtime dependencies.
    - `requirements-dev.txt` — runtime + test/dev dependencies.

---

## Project Structure
```markdown

mini\_rag/
├── src/
│   └── mini\_rag/
│       ├── app.py               # CLI + FastAPI entrypoint
│       ├── docs\_loader.py       # PDF/text loading + chunking
│       ├── embed.py / embedders # Embedding utilities
│       ├── indexer.py           # Embedding + vector index build
│       ├── ingest.py            # Document ingestion
│       ├── query\_cli.py         # CLI for queries
│       ├── retrieval.py         # Retrieval helpers
│       ├── utils.py             # Utility functions
│       └── tests/               # pytest tests
│           ├── test\_docs\_loader.py
│           ├── test\_indexer.py
│           ├── test\_pipeline.py
│           └── conftest.py
├── notebooks/
│   └── demo.md
├── requirements.txt             # runtime dependencies
├── requirements-dev.txt         # runtime + dev/test deps
├── run\_sanity.py                # basic sanity check script
└── .github/workflows/ci.yml     # GitHub Actions CI config
```


---

## Quick start (local)

1. **Install runtime dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **(Optional) Dev setup with tests**

   ```bash
   python -m pip install -r requirements-dev.txt
   ```

3. **Run CLI**

   ```bash
   python -m mini_rag.app
   ```

4. **Run FastAPI server**

   ```bash
   uvicorn src.mini_rag.app:app --reload
   ```

5. **Run tests**

   ```bash
   pytest -n auto --maxfail=1 --showlocals --durations=10
   ```

---

## Notes

* On Windows, `faiss-cpu` may be tricky to install; if missing, a NumPy-based fallback indexer is used.
* Large PDFs are streamed into overlapping chunks using the safe `chunk_text` implementation.
* `PyPDF2` is deprecated → this project uses `pypdf`.

---

## Roadmap

* Add coverage reports to CI.
* Expand FastAPI endpoints with file upload support.
* Experiment with hybrid indexes (BM25 + embeddings).
* Optional Dockerfile for reproducible setup.
