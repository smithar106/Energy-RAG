"""Source attribution: build the citation list from retrieved evidence.

The final answer references sources as ``[n]``. This module maps those markers
back to real chunks/documents so the response carries a verifiable trail.
"""
from __future__ import annotations

import re

from app.schemas.api import Source


def build_citations(evidence: list[dict]) -> list[Source]:
    """Return a deduplicated, ordered list of cited sources."""
    sources: list[Source] = []
    seen: set[int] = set()
    for chunk in evidence:
        cid = chunk.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        excerpt = (chunk.get("text") or "")[:240]
        sources.append(
            Source(
                id=cid,
                title=chunk.get("title", "untitled"),
                source=chunk.get("source", "unknown"),
                url=chunk.get("source_url"),
                excerpt=excerpt,
            )
        )
    return sources


def citation_markers(answer: str) -> set[int]:
    """Return the set of citation indices referenced as [n] in the answer."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
