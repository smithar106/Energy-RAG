"""Retrieval query construction.

The model may draft a semantic query, but the structured dimensions returned
from SQL (energy type, metric, geography, sector, time window, identified
event) are injected programmatically so retrieval is always anchored to the
actual series being discussed.
"""
from __future__ import annotations

from app.retrieval.series import series_meta
from app.retrieval.temporal import TimePeriod

# Driver concepts that explain electricity/energy price movements.
ELECTRICITY_CONCEPTS = (
    "electricity generation costs fuel costs natural gas coal "
    "electricity demand weather generation mix utility costs "
    "retail electricity prices"
)


def _series_from_sql(sql_results: list[dict]) -> str | None:
    for res in sql_results or []:
        change = res.get("price_change")
        if change and change.get("series_id"):
            return change["series_id"]
        changes = res.get("changes")
        if changes:
            return changes[0].get("series_id")
        rows = res.get("rows") or []
        if rows and rows[0].get("series_id"):
            return rows[0]["series_id"]
    return None


def _event_from_sql(sql_results: list[dict]) -> str | None:
    for res in sql_results or []:
        changes = res.get("changes")
        if changes:
            c = changes[0]
            prev = (c.get("previous_period") or "")[:7]
            curr = (c.get("current_period") or "")[:7]
            label = f"largest month-over-month {res.get('direction', 'increase')} {prev} to {curr}"
            return label
    return None


def build_retrieval_query(
    semantic_query: str,
    sql_results: list[dict],
    period: TimePeriod,
) -> str:
    """Compose the final retrieval query from the model query + SQL dimensions."""
    parts: list[str] = []
    if semantic_query:
        parts.append(semantic_query.strip())

    series_id = _series_from_sql(sql_results)
    meta = series_meta(series_id)

    dims: list[str] = []
    if meta.energy_type:
        dims.append(meta.energy_type)
    if meta.metric:
        dims.append(meta.metric)
    if meta.geography:
        dims.append(meta.geography)
    if meta.sector:
        dims.append(meta.sector)

    if period.is_bounded:
        ys = period.start.year if period.start else period.end.year
        ye = period.end.year if period.end else ys
        dims.append(str(ys) if ys == ye else f"{ys} {ye}")
        dims.append(f"context {ys - 1} to {ye + 1}")

    event = _event_from_sql(sql_results)
    if event:
        dims.append(event)

    if meta.energy_type in (None, "electricity"):
        dims.append(ELECTRICITY_CONCEPTS)

    return " ".join(p for p in parts if p) + " " + " ".join(dims)
