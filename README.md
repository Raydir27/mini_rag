# mini_rag
A small RAG-based Q&amp;A web app (local) where a user uploads one or more PDFs / text files and asks questions.

# mini-rag-day1

Day 1 deliverable for mini-RAG internship project.

## What this does
- Extracts text from a PDF.
- Splits into passages.
- Generates embeddings using `sentence-transformers`.
- Builds a FAISS index and provides a simple CLI to query it.

## Quickstart (local)
1. Create a venv and install:
   ```bash
   python -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
