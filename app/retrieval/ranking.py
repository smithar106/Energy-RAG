"""Hybrid ranking: semantic similarity + temporal relevance + source authority.

The score is explicit and inspectable — no opaque framework:

    final_score = W_semantic * semantic_similarity
                + W_temporal * temporal_score
                + W_authority * source_authority

Component definitions
---------------------
semantic_similarity : cosine similarity from pgvector, in [0, 1].
temporal_score      : 0.5 when the query has no period or the chunk has no event
                      window (neutral); 0.5–1.0 when the chunk's event window
                      overlaps the query period (scaled by coverage); decaying
                      to 0 as the gap between windows grows.
source_authority    : EIA analysis = 1.0, Wikipedia = 0.6, unknown = 0.5.
                      The EIA boost breaks ties toward the authoritative source.

Every ranked record carries its component scores so the ranking can be shown in
the RAG trace.
"""
from __future__ import annotations

from app.retrieval.temporal import TimePeriod

DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.60,
    "temporal": 0.25,
    "authority": 0.15,
}

RANKING_FORMULA = (
    "final_score = 0.60*semantic_similarity "
    "+ 0.25*temporal_score "
    "+ 0.15*source_authority"
)

SOURCE_AUTHORITY: dict[str, float] = {
    "U.S. Energy Information Administration": 1.0,
    "EIA": 1.0,
    "Wikipedia": 0.6,
}
DEFAULT_AUTHORITY = 0.5


def authority_for(source_name: str | None) -> float:
    if not source_name:
        return DEFAULT_AUTHORITY
    return SOURCE_AUTHORITY.get(source_name, DEFAULT_AUTHORITY)


def temporal_score(
    query_start_year: int | None,
    query_end_year: int | None,
    chunk_start_year: int | None,
    chunk_end_year: int | None,
) -> tuple[float, str]:
    """Return (score, reason) for the temporal component."""
    if query_start_year is None and query_end_year is None:
        return 0.5, "no_query_period"

    qs = query_start_year if query_start_year is not None else 1900
    qe = query_end_year if query_end_year is not None else 2100

    if chunk_start_year is None and chunk_end_year is None:
        return 0.5, "no_event_window"

    cs = chunk_start_year if chunk_start_year is not None else qs
    ce = chunk_end_year if chunk_end_year is not None else qe

    overlap = min(qe, ce) - max(qs, cs) + 1
    if overlap > 0:
        query_span = max(1, qe - qs + 1)
        score = 0.5 + 0.5 * min(1.0, overlap / query_span)
        return round(score, 4), "overlap"

    gap = min(abs(qs - ce), abs(cs - qe))
    score = max(0.0, 0.5 - 0.1 * gap)
    return round(score, 4), "gap"


def rank_chunks(
    candidates: list[dict],
    *,
    period: TimePeriod,
    top_n: int,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Score and re-sort candidates; return the top-N."""
    weights = weights or DEFAULT_WEIGHTS
    qs = period.start.year if period.start else None
    qe = period.end.year if period.end else None

    ranked: list[dict] = []
    for candidate in candidates:
        semantic = float(candidate.get("similarity", 0.0))
        temporal, reason = temporal_score(
            qs, qe, candidate.get("start_year"), candidate.get("end_year")
        )
        authority = authority_for(candidate.get("source_name"))
        final = (
            weights["semantic"] * semantic
            + weights["temporal"] * temporal
            + weights["authority"] * authority
        )
        record = dict(candidate)
        record.update(
            {
                "semantic_similarity": round(semantic, 4),
                "temporal_score": temporal,
                "temporal_reason": reason,
                "authority_score": authority,
                "final_score": round(final, 4),
            }
        )
        ranked.append(record)

    ranked.sort(key=lambda r: r["final_score"], reverse=True)
    return ranked[:top_n]
