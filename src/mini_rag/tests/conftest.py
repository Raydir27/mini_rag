# src/mini_rag/tests/conftest.py
import numpy as np
import pytest

class DummyModel:
    def __init__(self, dim=32):
        self.dim = dim

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        embs = []
        for t in texts:
            h = abs(hash(t))
            rng = np.random.default_rng(h % (2**32))
            vec = rng.standard_normal(self.dim).astype("float32")
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            embs.append(vec)
        return np.vstack(embs)

@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    """
    Patch sentence_transformers.SentenceTransformer to a fast DummyModel for pytest.
    """
    def _fake_ctor(model_name):
        return DummyModel(dim=32)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _fake_ctor, raising=False)
    # also patch module attribute if imported as module
    try:
        import sentence_transformers as st
        monkeypatch.setattr(st, "SentenceTransformer", _fake_ctor, raising=False)
    except Exception:
        pass

    yield
