"""Semantic chunking.

Splits heading-scoped segments into chunks that respect paragraph and sentence
boundaries, targeting ~500–800 tokens with ~100 tokens of overlap.

Token counts are approximated with words (1 word ≈ 0.75 English tokens), so the
defaults below land inside the target band:

    target_words = 550  ≈ 730 tokens
    overlap_words = 80  ≈ 106 tokens
    max_words = 800     ≈ 1060 tokens (hard ceiling for a single paragraph split)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.html import Segment

TARGET_WORDS = 550
OVERLAP_WORDS = 80
MAX_WORDS = 800

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    index: int
    section: str | None


def _split_long_text(text: str, target_words: int, overlap_words: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries."""
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    pieces: list[str] = []
    buf: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if buf and len(buf) + len(words) > target_words:
            pieces.append(" ".join(buf))
            buf = buf[-overlap_words:] if overlap_words < len(buf) else []
        buf.extend(words)
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def chunk_segments(
    segments: list[Segment],
    *,
    target_words: int = TARGET_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    max_words: int = MAX_WORDS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_section: str | None = None

    def emit(with_overlap: bool = True) -> None:
        nonlocal buf
        if not buf:
            return
        text = " ".join(buf).strip()
        if text:
            chunks.append(Chunk(text=text, index=len(chunks), section=buf_section))
        if with_overlap and 0 < overlap_words < len(buf):
            buf = list(buf[-overlap_words:])
        else:
            buf = []

    for segment in segments:
        paragraphs = [p.strip() for p in segment.text.split("\n\n") if p.strip()]
        for paragraph in paragraphs:
            # Section change: flush without carrying overlap across the boundary.
            if segment.heading != buf_section and buf:
                emit(with_overlap=False)
            buf_section = segment.heading

            words = paragraph.split()
            if len(words) > max_words:
                if buf:
                    emit(with_overlap=False)
                for piece in _split_long_text(paragraph, target_words, overlap_words):
                    chunks.append(
                        Chunk(text=piece.strip(), index=len(chunks), section=segment.heading)
                    )
                continue

            if buf and len(buf) + len(words) > target_words:
                emit(with_overlap=True)
            buf.extend(words)

    emit(with_overlap=False)
    return chunks
