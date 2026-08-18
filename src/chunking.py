"""Split raw text into overlapping chunks.

Strategy: pack whole paragraphs up to chunk_size, so a chunk rarely cuts a
thought in half. Paragraphs longer than chunk_size fall back to sentence
packing, and a single sentence longer than chunk_size is hard-sliced.
Each chunk after the first is prefixed with the tail of the previous one
(overlap characters) so context spanning a boundary is not lost.
"""

from __future__ import annotations

import re

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _hard_slice(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    """Break an oversized paragraph on sentence boundaries."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_slice(sentence, size))
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= size:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, chunk_size: int = 800, overlap: int = 50) -> list[str]:
    """Return overlapping chunks of at most ~chunk_size characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    units: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > chunk_size:
            units.extend(_split_long_paragraph(paragraph, chunk_size))
        else:
            units.append(paragraph)

    # Pack units greedily into chunk_size buckets.
    packed: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                packed.append(current)
            current = unit
    if current:
        packed.append(current)

    if overlap == 0 or len(packed) < 2:
        return packed

    # Prepend the tail of the previous chunk to every chunk but the first.
    chunks = [packed[0]]
    for previous, chunk in zip(packed, packed[1:]):
        chunks.append(f"{previous[-overlap:].lstrip()}\n\n{chunk}")
    return chunks
