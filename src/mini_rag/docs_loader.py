# src/docs_loader.py
from typing import List, Iterator, Union
import pypdf as PyPDF

def load_pdf_text(pdf_path: str) -> str:
    text_pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return "\n\n".join(text_pages)


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    *,
    as_generator: bool = False
) -> Union[List[str], Iterator[str]]:
    """
    Split text into chunks of up to chunk_size characters (approx tokens here)
    with `overlap` characters overlapping between consecutive chunks.

    - Raises ValueError if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size.
    - If as_generator=True returns an iterator (safer for very large inputs).
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    n = len(text)
    if n == 0:
        return iter(()) if as_generator else []

    def _gen():
        idx = 0
        step = chunk_size - overlap  # guaranteed positive by checks above
        while idx < n:
            end = min(idx + chunk_size, n)
            yield text[idx:end].strip()
            if end == n:
                break
            idx += step

    return _gen() if as_generator else list(_gen())
