# src/docs_loader.py
from typing import List
import PyPDF2

def load_pdf_text(pdf_path: str) -> str:
    text_pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return "\n\n".join(text_pages)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of approx chunk_size tokens (characters here for simplicity)
    with overlap.
    """
    chunks = []
    idx = 0
    n = len(text)
    while idx < n:
        end = min(idx + chunk_size, n)
        chunks.append(text[idx:end].strip())
        idx = end - overlap
        if idx < 0:
            idx = 0
        if idx >= n:
            break
    # filter empties
    return [c for c in chunks if c]
