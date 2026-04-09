"""
PDF loader + text chunker.
Extracts text from a PDF page by page and splits it into overlapping chunks.
Includes a hard cap on characters processed to avoid OOM on large documents.
"""

import io
import logging
from typing import Generator, List

import pypdf

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 800    # characters per chunk (larger = fewer chunks = less RAM)
CHUNK_OVERLAP = 100   # overlap between consecutive chunks
MAX_CHARS     = 300_000   # ~200 pages; raise if you have more RAM


# ─── PDF text extraction ──────────────────────────────────────────────────────

def extract_text_from_pdf(file: io.BytesIO, max_chars: int = MAX_CHARS):
    """
    Extract text page by page, stopping once `max_chars` is reached.
    Returns (text, was_truncated).
    """
    reader    = pypdf.PdfReader(file)
    pages     = []
    total     = 0
    truncated = False

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        remaining = max_chars - total
        if len(text) >= remaining:
            pages.append(text[:remaining])
            truncated = True
            logger.warning(
                "PDF truncated at page %d (%d chars limit). "
                "Raise MAX_CHARS in document_loader.py to process more.",
                i + 1, max_chars,
            )
            break

        pages.append(text)
        total += len(text)

    full_text = "\n\n".join(pages)
    logger.info(
        "Extracted %d characters from %d pages (truncated=%s).",
        len(full_text), len(pages), truncated,
    )
    return full_text, truncated


# ─── Text chunker (generator — never holds all chunks in memory at once) ──────

def iter_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Generator[str, None, None]:
    """Yield overlapping text chunks without building a full list in memory."""
    if not text.strip():
        return

    start  = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Break at sentence boundary within the last 20 % of the window
        if end < length:
            boundary = text.rfind(". ", start + int(chunk_size * 0.8), end)
            if boundary != -1:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        next_start = end - overlap
        if next_start <= start:          # safety: always advance
            next_start = start + chunk_size
        start = next_start


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    return list(iter_chunks(text, chunk_size=chunk_size, overlap=overlap))


# ─── Combined loader ──────────────────────────────────────────────────────────

def load_pdf_chunks(
    file: io.BytesIO,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_chars: int = MAX_CHARS,
) -> tuple[List[str], bool]:
    """
    Returns (chunks, was_truncated).
    Caller should warn the user when was_truncated is True.
    """
    text, truncated = extract_text_from_pdf(file, max_chars=max_chars)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    logger.info("Produced %d chunks (truncated=%s).", len(chunks), truncated)
    return chunks, truncated
