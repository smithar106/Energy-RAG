"""Chunking: split clean text into fixed-size, overlapping windows."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Split ``text`` into ~``chunk_size``-char windows with ``overlap``.

    Tries to break on sentence boundaries so chunks stay semantically coherent.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Chunk(text=text, index=0)]

    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    chunks: list[Chunk] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # Carry overlap: keep the tail of the previous window.
            current = current[-overlap:] + " " + sentence if current else sentence
    if current.strip():
        chunks.append(current.strip())

    return [Chunk(text=c, index=i) for i, c in enumerate(chunks)]
