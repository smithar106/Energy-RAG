"""Tool implementations executed by the orchestrator.

Each tool returns a JSON-serializable result. These are the *only* sources of
ground truth the agent is allowed to use (SQL for numbers, pgvector for
explanation).
"""
from __future__ import annotations

import json

from app.config import get_settings
from app.retrieval.reranker import rerank
from app.retrieval.sql import deterministic_calculation, structured_price_query
from app.retrieval.temporal import parse_time_period
from app.retrieval.vector import vector_search


def _period_from_arg(arg: str | None):
    if arg:
        return parse_time_period(arg)
    from app.retrieval.temporal import TimePeriod

    return TimePeriod()


def tool_structured_price_lookup(args: dict) -> dict:
    series_id = args.get("series_id") or None
    period = _period_from_arg(args.get("period"))
    return structured_price_query(
        question="",
        period=period,
        series_id=series_id,
    )


def tool_deterministic_calculation(args: dict) -> dict:
    period = _period_from_arg(args.get("period"))
    return deterministic_calculation(
        operation=args["operation"],
        period=period,
        series_id=args.get("series_id") or None,
    )


def tool_evidence_search(args: dict) -> dict:
    settings = get_settings()
    period = _period_from_arg(args.get("period"))
    results = vector_search(
        args["query"],
        period=period,
        top_k=settings.retrieval_top_k,
    )
    ranked = rerank(results, retrieval_query=args["query"], period=period)
    return {"chunks": ranked, "count": len(ranked)}


TOOL_DISPATCH = {
    "structured_price_lookup": tool_structured_price_lookup,
    "deterministic_calculation": tool_deterministic_calculation,
    "evidence_search": tool_evidence_search,
}


def execute_tool(name: str, arguments: str) -> dict:
    args = json.loads(arguments or "{}")
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(args)
    except Exception as exc:  # surface tool errors to the agent for retry
        return {"error": str(exc)}
