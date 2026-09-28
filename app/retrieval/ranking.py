"""Hybrid ranking: semantic + temporal + source authority, with diversity.

The score is explicit and inspectable — no opaque framework:

    final_score = W_semantic * semantic_similarity
                + W_temporal * temporal_score
                + W_authority * source_authority

Component definitions
---------------------
semantic_similarity : cosine similarity from pgvector, in [0, 1].
temporal_score      : 0.5 when the query has no period or the chunk has no event
                      window; otherwise 0.5 + 0.5 * coverage * specificity,
                      which strongly prefers a chunk whose event window is as
                      *specific* as the query. A document narrowly about 2017
                      (span 1) scores far above one broadly tagged 2007–2020.
source_authority    : EIA = 1.0, DOE/government = 0.9, other = 0.6, Wikipedia
                      = 0.4 (supplementary).

After ranking, ``select_evidence`` applies:
  - an evidence threshold (chunks below it are rejected → refusal path), and
  - document diversity (one chunk per document first, then up to
    ``max_per_doc`` per document, so the final evidence spans several sources).
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
    "Energy Information Administration": 1.0,
    "EIA": 1.0,
    "U.S. Department of Energy": 0.9,
    "DOE": 0.9,
    "Wikipedia": 0.4,
}
DEFAULT_AUTHORITY = 0.6

MIN_EVIDENCE_SCORE = 0.50     # below this a chunk is rejected
RELATIVE_FLOOR = 0.60         # for diversity, a new doc must reach best*floor
MAX_PER_DOCUMENT = 3          # cap chunks per document in the final evidence


def authority_for(source_name: str | None) -> float:
    if not source_name:
        return DEFAULT_AUTHORITY
    return SOURCE_AUTHORITY.get(source_name, DEFAULT_AUTHORITY)


def temporal_score(
    query_start_year: int | None,
    query_end_year: int | None,
    chunk_start_year: int | None,
    chunk_end_year: int | None,
) -> tuple[float, str, dict]:
    """Return (score, reason, components) for the temporal component."""
    if query_start_year is None and query_end_year is None:
        return 0.5, "no_query_period", {"coverage": 0.0, "specificity": 0.0}

    qs = query_start_year if query_start_year is not None else 1900
    qe = query_end_year if query_end_year is not None else 2100

    if chunk_start_year is None and chunk_end_year is None:
        return 0.5, "no_event_window", {"coverage": 0.0, "specificity": 0.0}

    cs = chunk_start_year if chunk_start_year is not None else qs
    ce = chunk_end_year if chunk_end_year is not None else qe

    query_span = max(1, qe - qs + 1)
    chunk_span = max(1, ce - cs + 1)
    overlap = min(qe, ce) - max(qs, cs) + 1

    if overlap <= 0:
        gap = min(abs(qs - ce), abs(cs - qe))
        score = max(0.0, 0.5 - 0.1 * gap)
        return round(score, 4), "gap", {
            "coverage": 0.0, "specificity": 0.0,
            "query_span": query_span, "chunk_span": chunk_span, "gap": gap,
        }

    coverage = min(1.0, overlap / query_span)
    specificity = min(1.0, query_span / chunk_span)
    score = 0.5 + 0.5 * coverage * specificity
    return round(score, 4), "overlap", {
        "coverage": round(coverage, 3),
        "specificity": round(specificity, 3),
        "query_span": query_span,
        "chunk_span": chunk_span,
        "overlap": overlap,
    }


def rank_chunks(
    candidates: list[dict],
    *,
    period: TimePeriod,
    top_n: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Score and sort candidates by the hybrid score (no truncation if top_n=None)."""
    weights = weights or DEFAULT_WEIGHTS
    qs = period.start.year if period.start else None
    qe = period.end.year if period.end else None

    ranked: list[dict] = []
    for candidate in candidates:
        semantic = float(candidate.get("similarity", 0.0))
        temporal, reason, meta = temporal_score(
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
                "temporal_meta": meta,
                "authority_score": authority,
                "final_score": round(final, 4),
            }
        )
        ranked.append(record)

    ranked.sort(key=lambda r: r["final_score"], reverse=True)
    return ranked[:top_n] if top_n is not None else ranked


def select_evidence(
    ranked: list[dict],
    top_n: int,
    *,
    min_score: float = MIN_EVIDENCE_SCORE,
    relative_floor: float = RELATIVE_FLOOR,
    max_per_doc: int = MAX_PER_DOCUMENT,
) -> tuple[list[dict], list[dict]]:
    """Return (accepted, rejected) applying threshold + document diversity."""
    if not ranked:
        return [], []
    best = ranked[0]["final_score"]

    accepted: list[dict] = []
    rejected: list[dict] = []
    counts: dict = {}
    chosen: set = set()

    def acceptable(c: dict) -> bool:
        return c["final_score"] >= min_score and c["final_score"] >= best * relative_floor

    # Pass 1: at most one chunk per document (document diversity first).
    for c in ranked:
        if len(accepted) >= top_n:
            break
        doc = c.get("document_id")
        if doc in counts:
            continue
        if not acceptable(c):
            continue
        accepted.append(c)
        counts[doc] = 1
        chosen.add(c["id"])

    # Pass 2: fill remaining slots up to max_per_doc.
    for c in ranked:
        if len(accepted) >= top_n:
            break
        if c["id"] in chosen:
            continue
        doc = c.get("document_id")
        if counts.get(doc, 0) >= max_per_doc:
            continue
        if not acceptable(c):
            continue
        accepted.append(c)
        counts[doc] = counts.get(doc, 0) + 1
        chosen.add(c["id"])

    rejected = [c for c in ranked if c["id"] not in chosen]
    return accepted, rejected
