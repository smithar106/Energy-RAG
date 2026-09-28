"""Tool implementations executed by the orchestrator.

Every tool result is either SQL-derived data (numbers, with explicit
observation pairs) or pgvector-retrieved evidence. Nothing is model-generated.

The evidence tool's private ``_trace`` payload (query embedding, ranking
scores, accept/reject state) is consumed by the orchestrator and stripped before
the result is sent to the LLM.
"""
from __future__ import annotations

import json

from app.retrieval.changes import largest_changes
from app.retrieval.query_builder import build_retrieval_query
from app.retrieval.service import retrieve
from app.retrieval.sql import default_series_id, deterministic_calculation, structured_price_query
from app.retrieval.temporal import TimePeriod, parse_time_period

# Safe to expose to the LLM (no URLs → cannot invent citations; no embeddings).
_LLM_CHUNK_FIELDS = (
    "id",
    "document_title",
    "section",
    "source_name",
    "source_type",
    "published_date",
    "start_year",
    "end_year",
    "semantic_similarity",
    "temporal_score",
    "authority_score",
    "final_score",
    "text",
)


def _period_from_arg(arg: str | None) -> TimePeriod:
    return parse_time_period(arg) if arg else TimePeriod()


def _period_dict(period: TimePeriod) -> dict:
    return {
        "start": period.start.isoformat() if period.start else None,
        "end": period.end.isoformat() if period.end else None,
    }


def _shape_for_llm(chunks: list[dict]) -> list[dict]:
    return [{k: c.get(k) for k in _LLM_CHUNK_FIELDS} for c in chunks]


def tool_structured_price_lookup(args: dict) -> dict:
    return structured_price_query(
        question="",
        period=_period_from_arg(args.get("period")),
        series_id=args.get("series_id") or None,
    )


def tool_deterministic_calculation(args: dict) -> dict:
    return deterministic_calculation(
        operation=args["operation"],
        period=_period_from_arg(args.get("period")),
        series_id=args.get("series_id") or None,
    )


def tool_find_largest_change(args: dict) -> dict:
    series_id = args.get("series_id") or default_series_id()
    if not series_id:
        raise ValueError("no price series available")
    period = _period_from_arg(args.get("period"))
    metric = args.get("metric") or "percent"
    direction = args.get("direction") or "increase"
    changes = largest_changes(series_id, period, metric=metric, direction=direction)
    return {
        "series_id": series_id,
        "metric": metric,
        "direction": direction,
        "interpretation": f"largest month-over-month {metric} {direction}",
        "changes": [c.to_dict() for c in changes],
    }


def tool_evidence_search(args: dict, sql_results: list[dict] | None = None) -> dict:
    semantic_query = args["query"]
    period = _period_from_arg(args.get("period"))
    enriched_query = build_retrieval_query(semantic_query, sql_results or [], period)

    result = retrieve(enriched_query, period)
    public = {
        "retrieval_query": result.retrieval_query,
        "requested_period": _period_dict(result.period),
        "candidate_count": result.candidate_count,
        "accepted_count": len(result.ranked),
        "chunks": _shape_for_llm(result.ranked),
    }
    trace = {
        "retrieval_query": result.retrieval_query,
        "requested_period": _period_dict(result.period),
        "query_embedding": result.query_embedding,
        "embedding_dim": len(result.query_embedding),
        "candidate_count": result.candidate_count,
        "accepted_count": len(result.ranked),
        "top_n": result.top_n,
        "weights": result.weights,
        "formula": result.formula,
        "chunks": result.ranked_all,
    }
    public["_trace"] = trace
    return public


TOOL_DISPATCH = {
    "structured_price_lookup": tool_structured_price_lookup,
    "deterministic_calculation": tool_deterministic_calculation,
    "find_largest_change": tool_find_largest_change,
    "evidence_search": tool_evidence_search,
}


def execute_tool(name: str, arguments: str, sql_results: list[dict] | None = None) -> dict:
    args = json.loads(arguments or "{}")
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        if name == "evidence_search":
            return fn(args, sql_results)
        return fn(args)
    except Exception as exc:
        return {"error": str(exc)}
