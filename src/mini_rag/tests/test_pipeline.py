# tests/test_pipeline.py
import os
from mini_rag.docs_loader import load_pdf_text, chunk_text

def test_load_and_chunk():
    # basic smoke test: will not error even if path missing; user should place the file for full test
    sample_text = "This is a test. " * 50
    chunks = chunk_text(sample_text, chunk_size=50, overlap=10)
    chunks = list(chunks)  # Ensure chunks is a list
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, str)
