"""Tool implementations executed by the orchestrator.

Every tool result is either SQL-derived data (with explicit observation pairs)
or retrieved evidence. The evidence tool runs the three-stage retrieval
(vector + lexical → rerank → evidence gate) and returns the accepted chunks to
the LLM, while attaching a private ``_trace`` (scores, gate decisions) for the
RAG trace.
"""
from __future__ import annotations

import json

from app.retrieval.changes import largest_changes
from app.retrieval.service import retrieve
from app.retrieval.sql import default_series_id, deterministic_calculation, structured_price_query
from app.retrieval.temporal import TimePeriod, parse_time_period

# Safe to expose to the LLM (no URLs → cannot invent citations; no embeddings).
_LLM_CHUNK_FIELDS = (
    "id", "document_title", "section", "source_name", "source_type",
    "published_date", "start_year", "end_year", "energy_type", "market_layer",
    "final_score", "text",
)


def _period_from_arg(arg: str | None) -> TimePeriod:
    return parse_time_period(arg) if arg else TimePeriod()


def _period_dict(period: TimePeriod) -> dict:
    return {
        "start": period.start.isoformat() if period.start else None,
        "end": period.end.isoformat() if period.end else None,
    }


def _intent_dict(intent) -> dict:
    return {
        "energy_type": intent.energy_type,
        "market_layer": intent.market_layer,
        "geography": intent.geography,
        "sector": intent.sector,
        "event_start": intent.event_start.isoformat() if intent.event_start else None,
        "event_end": intent.event_end.isoformat() if intent.event_end else None,
        "is_causal": intent.is_causal,
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


def tool_evidence_search(
    args: dict,
    sql_results: list[dict] | None,
    question: str,
) -> dict:
    period = _period_from_arg(args.get("period"))
    semantic_query = args["query"]

    result = retrieve(
        question=question,
        semantic_query=semantic_query,
        sql_results=sql_results or [],
        period=period,
    )

    public = {
        "retrieval_query": semantic_query,
        "candidate_count": result.candidate_count,
        "accepted_count": len(result.ranked),
        "insufficient_evidence": len(result.ranked) == 0,
        "chunks": _shape_for_llm(result.ranked),
    }
    trace = {
        "intent": _intent_dict(result.intent),
        "queries": result.queries,
        "query_embedding": result.query_embedding,
        "embedding_dim": len(result.query_embedding),
        "vector_count": result.vector_count,
        "lexical_count": result.lexical_count,
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


def execute_tool(
    name: str,
    arguments: str,
    sql_results: list[dict] | None = None,
    question: str = "",
) -> dict:
    args = json.loads(arguments or "{}")
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        if name == "evidence_search":
            return fn(args, sql_results, question)
        return fn(args)
    except Exception as exc:
        return {"error": str(exc)}
