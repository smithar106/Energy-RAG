"""Data-aware retrieval query expansion.

Generates MULTIPLE retrieval queries from the structured intent + semantic
query (never hard-coded for a specific year), so vector + lexical search cover
different facets of the question (event, fuel costs, demand, weather, mix).
"""
from __future__ import annotations

from datetime import date

from app.retrieval.intent import RetrievalIntent

_MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
    12: "December",
}

ELECTRICITY_CONCEPTS = (
    "electricity generation costs fuel costs natural gas coal "
    "electricity demand weather generation mix utility costs"
)


def _event_label(intent: RetrievalIntent) -> str | None:
    if intent.event_start and intent.event_end:
        sm = _MONTHS[intent.event_start.month]
        em = _MONTHS[intent.event_end.month]
        if intent.event_start.year == intent.event_end.year:
            if sm == em:
                return f"{sm} {intent.event_start.year}"
            return f"{sm} {em} {intent.event_start.year}"
        return f"{sm} {intent.event_start.year} to {em} {intent.event_end.year}"
    return None


def build_retrieval_queries(
    semantic_query: str,
    intent: RetrievalIntent,
) -> list[str]:
    """Return a list of retrieval queries for the intent."""
    queries: list[str] = []
    if semantic_query:
        queries.append(semantic_query.strip())

    energy = intent.energy_type or "energy"
    metric = (intent.market_layer or "price").replace("_", " ")
    geo = intent.geography or ""
    year = intent.event_start.year if intent.event_start else None
    event = _event_label(intent)

    def join(*parts: str) -> str:
        return " ".join(p for p in parts if p)

    if event and year:
        queries.append(join(energy, metric, geo, event, "increase"))
        queries.append(join(geo, energy, "prices", event, "generation fuel costs"))
        queries.append(join("EIA", energy, "prices", event, "natural gas generation coal"))
        queries.append(join(str(year), geo, energy, "demand weather generation costs"))
        queries.append(join(str(year), energy, metric, "increase EIA drivers"))
    elif year:
        queries.append(join(energy, metric, geo, str(year), "increase"))
        queries.append(join(geo, energy, "prices", str(year), "generation fuel costs"))
        queries.append(join(str(year), energy, "demand weather generation costs"))

    if energy == "electricity":
        for q in list(queries):
            if "natural gas" not in q and "generation costs" not in q:
                queries.append(join(q, ELECTRICITY_CONCEPTS))

    # Dedupe, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out
