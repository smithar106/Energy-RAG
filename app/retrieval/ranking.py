"""Composite reranking + evidence gate.

Composite score (inspectable, not an opaque framework):

    final = w_sem*semantic + w_lex*lexical + w_temp*temporal
          + w_dom*domain + w_met*metric + w_geo*geography + w_auth*authority

The **evidence gate** then applies HARD, absolute filters (independent of how
bad the other candidates are): energy-domain mismatch, geography mismatch, and
— for causal questions with a specific event — temporal relevance. A chunk that
fails the gate is rejected even if its composite score is relatively high.
"""
from __future__ import annotations

import re
from datetime import date

from app.retrieval.intent import RetrievalIntent

DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.28,
    "lexical": 0.18,
    "temporal": 0.16,
    "domain": 0.14,
    "metric": 0.06,
    "geography": 0.06,
    "authority": 0.12,
}

RANKING_FORMULA = (
    "final = 0.28*semantic + 0.18*lexical + 0.16*temporal + 0.14*domain "
    "+ 0.06*metric + 0.06*geography + 0.12*authority"
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

# Hard energy-type mismatches for gating (candidate retrieval already filters,
# but the gate is authoritative).
MISMATCH_ENERGY = {
    "electricity": {"gasoline", "petroleum"},
    "natural_gas": {"gasoline"},
    "petroleum": {"electricity", "gasoline"},
    "gasoline": {"electricity", "coal"},
    "coal": {"gasoline", "petroleum"},
}


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
    """Year-level temporal relevance (coverage × specificity)."""
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
        return round(max(0.0, 0.5 - 0.1 * gap), 4), "gap", {
            "coverage": 0.0, "specificity": 0.0,
            "query_span": query_span, "chunk_span": chunk_span, "gap": gap,
        }

    coverage = min(1.0, overlap / query_span)
    specificity = min(1.0, query_span / chunk_span)
    return round(0.5 + 0.5 * coverage * specificity, 4), "overlap", {
        "coverage": round(coverage, 3), "specificity": round(specificity, 3),
        "query_span": query_span, "chunk_span": chunk_span, "overlap": overlap,
    }


def _lexical_overlap(query_text: str, text: str) -> float:
    terms = {t for t in re.findall(r"[a-z0-9]{2,}", (query_text or "").lower())}
    if not terms:
        return 0.0
    ttext = (text or "").lower()
    return sum(1 for t in terms if t in ttext) / len(terms)


def _energy_domain_score(intent: RetrievalIntent, chunk: dict) -> float:
    et = intent.energy_type
    cet = chunk.get("energy_type")
    if not et:
        return 0.5
    if not cet:
        return 0.5
    if cet == et:
        return 1.0
    if cet == "multi_energy":
        return 0.7
    if et == "electricity" and cet in {"natural_gas", "coal"}:
        return 0.7
    if et == "natural_gas" and cet == "electricity":
        return 0.6
    return 0.0


def _metric_score(intent: RetrievalIntent, chunk: dict) -> float:
    m = intent.market_layer
    cm = chunk.get("market_layer")
    if not m:
        return 0.5
    if not cm:
        return 0.5
    if cm == m:
        return 1.0
    if m == "retail_price" and cm in {"fuel_cost", "generation", "wholesale_price"}:
        return 0.6
    return 0.3


def _geography_score(intent: RetrievalIntent, chunk: dict) -> float:
    g = intent.geography
    cg = chunk.get("geography")
    if not g:
        return 0.5
    if not cg:
        return 0.5
    return 1.0 if cg == g else 0.0


def rank_chunks(
    candidates: list[dict],
    *,
    intent: RetrievalIntent,
    query_text: str,
    top_n: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    weights = weights or DEFAULT_WEIGHTS
    qs = intent.event_start.year if intent.event_start else None
    qe = intent.event_end.year if intent.event_end else None

    ranked: list[dict] = []
    for candidate in candidates:
        semantic = float(candidate.get("similarity", 0.0))
        lexical = _lexical_overlap(query_text, candidate.get("text", ""))
        temporal, t_reason, t_meta = temporal_score(
            qs, qe, candidate.get("start_year"), candidate.get("end_year")
        )
        domain = _energy_domain_score(intent, candidate)
        metric = _metric_score(intent, candidate)
        geography = _geography_score(intent, candidate)
        authority = authority_for(candidate.get("source_name"))
        final = (
            weights["semantic"] * semantic
            + weights["lexical"] * lexical
            + weights["temporal"] * temporal
            + weights["domain"] * domain
            + weights["metric"] * metric
            + weights["geography"] * geography
            + weights["authority"] * authority
        )
        record = dict(candidate)
        record.update(
            {
                "semantic_similarity": round(semantic, 4),
                "lexical_score": round(lexical, 4),
                "temporal_score": temporal,
                "temporal_reason": t_reason,
                "temporal_meta": t_meta,
                "domain_score": round(domain, 4),
                "metric_score": round(metric, 4),
                "geography_score": round(geography, 4),
                "authority_score": authority,
                "final_score": round(final, 4),
            }
        )
        ranked.append(record)

    ranked.sort(key=lambda r: r["final_score"], reverse=True)
    return ranked[:top_n] if top_n is not None else ranked


def _chunk_window(chunk: dict) -> tuple[date | None, date | None]:
    es = chunk.get("event_start_date")
    ee = chunk.get("event_end_date")
    if isinstance(es, str):
        es = date.fromisoformat(es)
    if isinstance(ee, str):
        ee = date.fromisoformat(ee)
    if es is None and chunk.get("start_year"):
        es = date(int(chunk["start_year"]), 1, 1)
    if ee is None and chunk.get("end_year"):
        ee = date(int(chunk["end_year"]), 12, 31)
    return es, ee


def gate_chunks(
    ranked: list[dict],
    *,
    intent: RetrievalIntent,
) -> tuple[list[dict], list[dict]]:
    """Hard evidence gate. Returns (passed, rejected) with reasons."""
    passed: list[dict] = []
    rejected: list[dict] = []
    context_start = intent.context_start
    context_end = intent.context_end

    for c in ranked:
        reasons: list[str] = []

        # Domain mismatch.
        et = intent.energy_type
        cet = c.get("energy_type")
        if et and cet and cet in MISMATCH_ENERGY.get(et, set()):
            reasons.append("energy domain mismatch")

        # Geography mismatch.
        if intent.geography and c.get("geography") == "international":
            reasons.append("geography mismatch")

        # Temporal gate (causal questions with a specific event).
        if intent.is_causal and context_start and context_end:
            es, ee = _chunk_window(c)
            pub = c.get("published_date")
            if isinstance(pub, str):
                pub = date.fromisoformat(pub)
            relevant = False
            if es and ee:
                relevant = ee >= context_start and es <= context_end
            elif pub:
                relevant = context_start <= pub <= context_end
            if not relevant:
                reasons.append("no temporal relevance to event")
            elif es and ee and (ee.year - es.year + 1) > 6:
                # A very broad window must actually mention the target year.
                target_years = set(range(context_start.year, context_end.year + 1))
                mentioned = {int(y) for y in (c.get("mentioned_years") or [])}
                if not (mentioned & target_years):
                    reasons.append("event window too broad and never mentions the target year")

        if reasons:
            c = dict(c)
            c["gate"] = "FAIL"
            c["failure_reason"] = "; ".join(reasons)
            rejected.append(c)
        else:
            c = dict(c)
            c["gate"] = "PASS"
            c["failure_reason"] = None
            passed.append(c)

    return passed, rejected


def apply_diversity(
    passed: list[dict],
    top_n: int,
    *,
    max_per_doc: int = 3,
) -> list[dict]:
    """Diversify gate-passed chunks (one per document first, then fill)."""
    accepted: list[dict] = []
    counts: dict = {}
    chosen: set = set()
    for c in passed:
        if len(accepted) >= top_n:
            break
        doc = c.get("document_id")
        if doc in counts:
            continue
        accepted.append(c)
        counts[doc] = 1
        chosen.add(c["id"])
    for c in passed:
        if len(accepted) >= top_n:
            break
        if c["id"] in chosen:
            continue
        doc = c.get("document_id")
        if counts.get(doc, 0) >= max_per_doc:
            continue
        accepted.append(c)
        counts[doc] = counts.get(doc, 0) + 1
        chosen.add(c["id"])
    return accepted
