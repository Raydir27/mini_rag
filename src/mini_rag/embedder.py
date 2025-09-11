# embedder.py
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

# model choice (fast & small) -- change if you want a different model
MODEL_NAME = "all-MiniLM-L6-v2"

# Create the model once (will download the model the first time)
_model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts: List[str], batch_size: int = 32, normalize: bool = False) -> List[List[float]]:
    """
    Embed a list of texts using sentence-transformers.
    Returns a list of embeddings (plain Python lists) suitable for faiss (float32).
    """
    if not texts:
        return []

    # encode -> returns numpy array (num_texts x dim)
    embs = _model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=normalize)
    # ensure float32
    embs = embs.astype("float32")
    # return as Python lists (indexer expects list-of-lists) OR keep numpy if you prefer
    return [emb.tolist() for emb in embs]

def embed_texts_numpy(texts: List[str], batch_size: int = 32, normalize: bool = False) -> np.ndarray:
    """
    Alternate helper: returns a numpy array (N x dim), convenient when passing directly to faiss.
    """
    if not texts:
        dim = _model.get_sentence_embedding_dimension() or 0
        return np.zeros((0, dim), dtype="float32")
    embs = _model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=normalize)
    return embs.astype("float32")

def get_dim() -> int:
    return _model.get_sentence_embedding_dimension() or 0
