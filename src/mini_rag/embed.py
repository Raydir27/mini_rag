# src/embed.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good baseline

class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
