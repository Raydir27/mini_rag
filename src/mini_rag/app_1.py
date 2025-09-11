# File: src/mini_rag/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os

from .retrieval import load_index, search
from sentence_transformers import SentenceTransformer



app = FastAPI()

# Path to Day 2 index (can be overridden via env var)
INDEX_PATH = Path(os.environ.get('MINI_RAG_INDEX_PATH', Path(__file__).parent.parent / 'data' / 'index.npz'))

# Load index at startup
try:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index file not found at {INDEX_PATH}")
    INDEX_OBJ, _ = load_index(str(INDEX_PATH))
except Exception as e:
    INDEX_OBJ = None
    print(f"[WARN] Failed to load index: {e}")

# Load sentence-transformers model once
MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
try:
    ST_MODEL = SentenceTransformer(MODEL_NAME)
except Exception as e:
    ST_MODEL = None
    print(f"[WARN] Failed to load SentenceTransformer model: {e}")


class QueryReq(BaseModel):
    query: str
    top_k: int = 5


def embed_fn(text: str):
    if ST_MODEL is None:
        raise RuntimeError("SentenceTransformer model not loaded")
    return ST_MODEL.encode([text], convert_to_numpy=True)[0]


@app.post('/query')
async def query(req: QueryReq):
    if INDEX_OBJ is None:
        raise HTTPException(status_code=500, detail="Index not loaded. Check server logs.")

    try:
        results = search(INDEX_OBJ, embed_fn, req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return {"query": req.query, "results": results}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
# To run: uvicorn src.mini_rag.app:app --reload --host 0.0.0.0 --port 8000