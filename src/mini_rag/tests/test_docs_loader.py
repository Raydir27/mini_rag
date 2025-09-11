# src/mini_rag/tests/test_docs_loader.py
import math
import collections.abc
import pytest
from mini_rag.docs_loader import chunk_text

SAMPLE = "This is a test. " * 50  # length ~ 800

def _ensure_list(obj):
    # if it's an iterator or iterable but not sized, convert to list
    if isinstance(obj, collections.abc.Iterator) or not isinstance(obj, collections.abc.Sized):
        return list(obj)
    return obj

def test_chunk_text_basic():
    chunk_size = 50
    overlap = 10
    chunks = chunk_text(SAMPLE, chunk_size=chunk_size, overlap=overlap)
    chunks = _ensure_list(chunks)   # <- safe conversion
    step = chunk_size - overlap
    expected = math.ceil(len(SAMPLE) / step)
    assert len(chunks) == expected
    assert all(c for c in chunks)

def test_chunk_text_overlap_error():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=5, overlap=5)  # overlap == chunk_size invalid

def test_chunk_text_generator_consistent():
    chunk_size = 60
    overlap = 5
    gen = chunk_text(SAMPLE, chunk_size=chunk_size, overlap=overlap, as_generator=True)
    assert hasattr(gen, "__iter__")
    lst = list(gen)                # explicitly materialize generator
    assert len(lst) > 0
    lst2 = chunk_text(SAMPLE, chunk_size=chunk_size, overlap=overlap)
    # ensure lst2 is a list (function default returns list)
    assert lst == lst2
