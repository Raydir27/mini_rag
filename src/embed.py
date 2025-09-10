# src/embed.py
import os
import time
import math
from typing import List, Optional
import numpy as np
import openai

# Default model (cheap & accurate): text-embedding-3-small (1536 dims).
# See OpenAI docs for alternative models. You can override with EMBEDDING_MODEL env var.
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set. Export it (or set GitHub secret).")

openai.api_key = OPENAI_API_KEY

def _chunk_list(lst: List[str], chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def embed_texts(texts: List[str], model: Optional[str] = None, batch_size: int = 32) -> np.ndarray:
    """
    Create embeddings for a list of texts via OpenAI Embeddings API.
    Returns a numpy array of shape (len(texts), dim).
    - model: e.g. "text-embedding-3-small" or "text-embedding-3-large"
    - batch_size: how many inputs to send per request (OpenAI supports arrays)
    """
    model = model or DEFAULT_MODEL
    embeddings = []
    total = len(texts)
    # send in batches
    for batch in _chunk_list(texts, batch_size):
        # retry loop (simple)
        for attempt in range(5):
            try:
                resp = openai.Embedding.create(model=model, input=batch)
                # resp['data'] is a list matching batch order
                for item in resp['data']:
                    embeddings.append(item['embedding'])
                break
            except Exception as e:
                wait = (2 ** attempt) + 0.1 * attempt
                time.sleep(wait)
                if attempt == 4:
                    raise
    arr = np.array(embeddings, dtype=np.float32)
    # normalize for cosine-similarity with inner product index
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr
