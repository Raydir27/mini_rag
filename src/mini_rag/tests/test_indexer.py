# tests/test_indexer.py
from mini_rag.indexer import MiniIndexer

def test_build_and_search_basic():
    docs = ["This is a red apple.", "Bananas are yellow.", "I like apples and bananas."]
    ids = ["a.txt", "b.txt", "c.txt"]
    idx = MiniIndexer()
    idx.build(docs, ids)
    res = idx.search("apple", top_k=2)
    assert len(res) >= 1
    assert "apple" in res[0][1].lower()
