"""Evidence ranking (reranking/filtering).

Ranks retrieved chunks by a deterministic blend of:

  - vector cosine similarity (from pgvector)
  - temporal proximity to the requested period
  - lexical overlap with the query (a lightweight BM25-ish signal)

Returns the top-N chunks used to ground the final answer.
"""
from __future__ import annotations

import math
import re
from datetime import date

from app.config import get_settings
from app.retrieval.temporal import TimePeriod


def _lexical_overlap(query: str, text: str) -> float:
    q_terms = {t for t in re.findall(r"[a-z0-9]{3,}", query.lower())}
    if not q_terms:
        return 0.0
    t_terms = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    return len(q_terms & t_terms) / len(q_terms)


def _temporal_score(period: TimePeriod, start: date | None, end: date | None) -> float:
    if not period.is_bounded or (start is None and end is None):
        return 0.5  # neutral when no metadata or no period
    if start is None:
        start = date.min
    if end is None:
        end = date.max
    lo = period.start or date.min
    hi = period.end or date.max

    overlap_start = max(start, lo)
    overlap_end = min(end, hi)
    if overlap_start > overlap_end:
        return 0.0  # no temporal overlap → strong downweight

    span = (hi - lo).days or 1
    overlap = (overlap_end - overlap_start).days + 1
    # Overlap fraction mapped through sqrt to keep partial matches alive.
    return math.sqrt(min(overlap / span, 1.0))


def rerank(
    chunks: list[dict],
    *,
    retrieval_query: str,
    period: TimePeriod,
    top_n: int | None = None,
) -> list[dict]:
    """Score and re-sort chunks, returning the top-N."""
    settings = get_settings()
    top_n = top_n or settings.rerank_top_n

    scored = []
    for c in chunks:
        sim = float(c.get("similarity", 0.0))
        lex = _lexical_overlap(retrieval_query, c.get("text", ""))
        tmp = _temporal_score(period, c.get("start_date"), c.get("end_date"))
        # similarity dominant; lexical + temporal as tuned additives.
        score = sim + 0.15 * lex + settings.temporal_priority_weight * (tmp - 0.5)
        c = dict(c)
        c["_score"] = round(score, 5)
        scored.append(c)

    scored.sort(key=lambda c: c["_score"], reverse=True)
    return scored[:top_n]
