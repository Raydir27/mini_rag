# ingest.py
from pathlib import Path
from typing import List, Dict
import tiktoken


CHUNK_SIZE = 800
CHUNK_OVERLAP = 200




def read_text_files(path: str) -> List[Dict]:
    p = Path(path)
    docs = []
    for f in p.glob('**/*.txt'):
        text = f.read_text(encoding='utf-8')
        docs.append({'id': str(f.relative_to(p)), 'text': text, 'source': str(f)})
    return docs

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
# naive whitespace chunker that respects token boundaries roughly
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks



def prepare_documents(root: str) -> List[Dict]:
    raw = read_text_files(root)
    out = []
    for doc in raw:
        chunks = chunk_text(doc['text'])
        for idx, c in enumerate(chunks):
            out.append({
            'id': f"{doc['id']}::chunk_{idx}",
            'text': c,
            'source': doc['source'],
            'chunk_index': idx
            })
    return out




if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else './data'
    docs = prepare_documents(root)
    print(f'Prepared {len(docs)} document chunks')