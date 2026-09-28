"""Structured SQL retrieval + deterministic calculations.

This is the **quantitative truth** path. Every number reported to the user is
produced here by SQL against ``price_records`` — never by the LLM.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import text

from app.db.base import session_scope
from app.retrieval.temporal import TimePeriod


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
            sql = f"""
                WITH bounds AS (
                    SELECT
                        (array_agg(price ORDER BY period ASC))[1]  AS first_price,
                        (array_agg(price ORDER BY period DESC))[1] AS last_price
                    FROM price_records
                    WHERE 1=1 {frag} {series_clause}
                )
                SELECT first_price, last_price,
                       (last_price - first_price) AS change,
                       CASE WHEN first_price = 0 THEN NULL
                            ELSE ROUND(((last_price - first_price) / first_price * 100)::numeric, 2)
                       END AS pct_change
                FROM bounds
            """
            row = dict(session.execute(text(sql), params).mappings().one())
            if operation == "change":
                return {"operation": operation, "value": row["change"]}
            return {"operation": operation, "value": row["pct_change"]}

    raise ValueError(f"unknown operation: {operation}")
