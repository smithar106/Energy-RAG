"""Structured SQL retrieval + deterministic calculations.

This is the **quantitative truth** path. Every number reported to the user is
produced here by SQL against ``price_records`` — never by the LLM.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import text

from app.db.base import session_scope
from app.retrieval.temporal import TimePeriod


def default_series_id() -> str | None:
    """Return the most populated price series (fallback when none is given)."""
    with session_scope() as session:
        row = session.execute(
            text(
                "SELECT series_id FROM price_records "
                "GROUP BY series_id ORDER BY COUNT(*) DESC LIMIT 1"
            )
        ).first()
    return row[0] if row else None


def _period_fragment(period: TimePeriod) -> tuple[str, dict]:
    if period.start and period.end:
        return "AND period BETWEEN :t_start AND :t_end", {
            "t_start": period.start,
            "t_end": period.end,
        }
    if period.start:
        return "AND period >= :t_start", {"t_start": period.start}
    if period.end:
        return "AND period <= :t_end", {"t_end": period.end}
    return "", {}


def structured_price_query(
    question: str,
    period: TimePeriod,
    *,
    series_id: str | None = None,
) -> dict:
    """Return verified price statistics for the requested period.

    Returns the raw SQL, the sampled rows, and *deterministic* aggregates
    (avg/min/max/latest) computed in SQL — so the numbers are reproducible.
    """
    frag, params = _period_fragment(period)
    series_clause = "AND series_id = :series_id" if series_id else ""
    if series_id:
        params["series_id"] = series_id

    sql = f"""
        SELECT
            period, price, units, region, fuel, series_id
        FROM price_records
        WHERE 1=1 {frag} {series_clause}
        ORDER BY period ASC
    """

    with session_scope() as session:
        rows = [
            dict(row._mapping)
            for row in session.execute(text(sql), params)
        ]

        agg_sql = f"""
            SELECT
                COUNT(*)                       AS n,
                ROUND(AVG(price)::numeric, 4)  AS avg_price,
                ROUND(MIN(price)::numeric, 4)  AS min_price,
                ROUND(MAX(price)::numeric, 4)  AS max_price
            FROM price_records
            WHERE 1=1 {frag} {series_clause}
        """
        agg = dict(session.execute(text(agg_sql), params).mappings().one())

    derived = {
        "count": int(agg["n"] or 0),
        "average": float(agg["avg_price"]) if agg["avg_price"] is not None else None,
        "minimum": float(agg["min_price"]) if agg["min_price"] is not None else None,
        "maximum": float(agg["max_price"]) if agg["max_price"] is not None else None,
    }
    return {"query": sql, "rows": rows, "derived": derived}


def deterministic_calculation(
    operation: str,
    period: TimePeriod,
    *,
    series_id: str | None = None,
) -> dict:
    """Explicit deterministic math on top of the structured table.

    Operations: ``average``, ``minimum``, ``maximum``, ``change`` (latest minus
    earliest), ``pct_change`` (relative change over the period).
    """
    frag, params = _period_fragment(period)
    series_clause = "AND series_id = :series_id" if series_id else ""
    if series_id:
        params["series_id"] = series_id

    select_agg = {
        "average": "AVG(price)",
        "minimum": "MIN(price)",
        "maximum": "MAX(price)",
    }

    with session_scope() as session:
        if operation in select_agg:
            sql = f"SELECT ROUND({select_agg[operation]}::numeric, 4) AS v FROM price_records WHERE 1=1 {frag} {series_clause}"
            v = session.execute(text(sql), params).scalar()
            return {"operation": operation, "value": float(v) if v is not None else None}

    if operation in {"change", "pct_change"}:
        # Return the full observation pair so the two figures can never be
        # recombined with a change computed over a different pair.
        from app.retrieval.changes import endpoint_change

        if not series_id:
            raise ValueError("change calculations require a series_id")
        change = endpoint_change(series_id, period)
        if change is None:
            return {"operation": operation, "price_change": None}
        return {"operation": "period_end_to_end", "price_change": change.to_dict()}

    raise ValueError(f"unknown operation: {operation}")
