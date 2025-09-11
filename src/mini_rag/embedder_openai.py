#embedder.py
from typing import List
import os
import time


# Example using OpenAI embeddings - adapt if using HF or other providers
import openai
from sentence_transformers import SentenceTransformer


openai.api_key = os.getenv('OPENAI_API_KEY')
EMBED_MODEL = 'text-embedding-3-small' # change if needed




def embed_texts(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = openai.Embeddings.create(model=EMBED_MODEL, input=batch)
        embeds = [r['embedding'] for r in resp['data']]
        all_embeddings.extend(embeds)
        time.sleep(0.1)
    return all_embeddings