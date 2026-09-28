"""Source attribution.

Citations are injected **programmatically** from the retrieved chunk records —
the LLM never sees or emits URLs/titles/dates, so it cannot invent them.

The answer references evidence as ``[n]`` where ``n`` is the 0-based index into
the ranked evidence list. This module maps those indices back to the real
stored records (chunk id, document id, source URL, publication date, event
window) so every citation links to its actual source.
"""
from __future__ import annotations

import re

from app.schemas.api import Source

_CITE_RE = re.compile(r"\[(\d+)\]")


def build_citations(evidence: list[dict]) -> list[Source]:
    """Return citations in evidence order (index n ↔ answer marker [n])."""
    citations: list[Source] = []
    for index, chunk in enumerate(evidence):
        citations.append(
            Source(
                index=index,
                chunk_id=int(chunk.get("id")),
                document_id=int(chunk.get("document_id")),
                title=chunk.get("document_title") or "",
                source_name=chunk.get("source_name") or "unknown",
                source_type=chunk.get("source_type") or "unknown",
                url=chunk.get("source_url"),
                published_date=chunk.get("published_date"),
                start_year=chunk.get("start_year"),
                end_year=chunk.get("end_year"),
                section=chunk.get("section"),
                excerpt=(chunk.get("text") or "")[:280],
            )
        )
    return citations


def citation_markers(answer: str) -> list[int]:
    """Return citation indices referenced as [n] in the answer (in order)."""
    return [int(n) for n in _CITE_RE.findall(answer or "")]
