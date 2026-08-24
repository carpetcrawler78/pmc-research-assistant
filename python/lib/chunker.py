"""
lib/chunker.py
Capstone 1 — PMC Research Assistant  v2

Recursive character-based text chunker for BERT-compatible embedding.
BERT models (including S-PubMedBERT) have a 512-token limit.
Rule of thumb: 1 token ≈ 4 chars  →  CHUNK_MAX_CHARS = 1600 ≈ 400 tokens.

Public API
----------
chunk_text(text, max_chars, overlap_chars) -> list[str]
build_text(row, fields)                    -> str
"""

from __future__ import annotations

# Separators tried in order when splitting (coarsest → finest)
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(
    text: str,
    max_chars: int = 1600,
    overlap_chars: int = 200,
) -> list[str]:
    """
    Split *text* into overlapping chunks of at most *max_chars* characters.
    Tries to break at sentence boundaries before falling back to whitespace.

    Parameters
    ----------
    text          : input string (any length)
    max_chars     : hard upper bound on chunk size in characters
    overlap_chars : how many chars from the end of the previous chunk to
                    prepend to the next one (context continuity)

    Returns
    -------
    list of non-empty stripped strings; at least one element if text is non-empty.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            # Try to break at a natural boundary inside the window
            cut = -1
            for sep in _SEPARATORS:
                pos = text.rfind(sep, start + overlap_chars, end)
                if pos != -1:
                    cut = pos + len(sep)
                    break
            if cut != -1:
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Next window starts with overlap
        start = max(end - overlap_chars, start + 1)

    return chunks


def build_text(row: dict, fields: list[str], sep: str = " [SEP] ") -> str:
    """
    Concatenate *fields* from *row* dict into a single string.
    Missing / None / empty fields are silently skipped.

    Example
    -------
    build_text(row, ["title", "abstract", "introduction"])
    """
    parts = [str(row.get(f) or "").strip() for f in fields]
    return sep.join(p for p in parts if p)
