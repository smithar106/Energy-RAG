"""Deterministic price-change calculations with explicit observation pairs.

Every change is represented as a :class:`PriceChange` that preserves the exact
observation pair used. Absolute and percentage change are always derived from
the *same* pair, which prevents the model from combining statistics computed
over different periods (the 2017 bug).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from sqlalchemy import text

from app.db.base import session_scope
from app.retrieval.series import series_meta
from app.retrieval.temporal import TimePeriod


@dataclass
class PriceChange:
    calculation_type: str          # "month_over_month" | "period_end_to_end"
    series_id: str
    energy_type: str | None
    metric: str | None
    sector: str | None
    geography: str | None
    unit: str
    previous_period: str
    previous_value: float
    current_period: str
    current_value: float
    absolute_change: float
    percent_change: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _build_change(
    *,
    calculation_type: str,
    series_id: str,
    units: str,
    prev_period: date,
    prev_value: float,
    curr_period: date,
    curr_value: float,
) -> PriceChange:
    meta = series_meta(series_id)
    absolute = curr_value - prev_value
    percent = None if prev_value == 0 else (absolute / prev_value) * 100.0
    return PriceChange(
        calculation_type=calculation_type,
        series_id=series_id,
        energy_type=meta.energy_type,
        metric=meta.metric,
        sector=meta.sector,
        geography=meta.geography,
        unit=units or meta.unit or "",
        previous_period=prev_period.isoformat(),
        previous_value=round(prev_value, 4),
        current_period=curr_period.isoformat(),
        current_value=round(curr_value, 4),
        absolute_change=round(absolute, 4),
        percent_change=round(percent, 4) if percent is not None else None,
    )


def _period_clause(period: TimePeriod, params: dict, *, column: str = "period") -> str:
    clause = ""
    if period.start:
        params["t_start"] = period.start
        clause += f" AND {column} >= :t_start"
    if period.end:
        params["t_end"] = period.end
        clause += f" AND {column} <= :t_end"
    return clause


def largest_changes(
    series_id: str,
    period: TimePeriod,
    *,
    metric: str = "percent",
    direction: str = "increase",
    k: int = 5,
) -> list[PriceChange]:
    """Find the largest month-over-month changes via a LAG() window.

    ``metric``: "percent" | "absolute". ``direction``: "increase" | "decrease".
    A change is included when its *current* period falls inside ``period`` (so a
    January change still uses the previous December, even across the boundary).
    """
    order_col = "pct_change" if metric == "percent" else "abs_change"
    # NULL percent changes (zero previous) are never "largest".
    nulls = " NULLS LAST" if metric == "percent" else ""
    order_dir = "DESC" if direction == "increase" else "ASC"
    order_clause = f"{order_col} {order_dir}{nulls}"

    params: dict = {"sid": series_id, "k": k}
    current_clause = ""
    if period.start:
        params["t_start"] = period.start
        current_clause += " AND current_period >= :t_start"
    if period.end:
        params["t_end"] = period.end
        current_clause += " AND current_period <= :t_end"
    lookback = ""
    if period.end:
        lookback = "WHERE series_id = :sid AND period <= :t_end"
    else:
        lookback = "WHERE series_id = :sid"

    sql = f"""
        WITH ordered AS (
            SELECT
                period,
                price,
                units,
                LAG(period) OVER (ORDER BY period) AS prev_period,
                LAG(price)  OVER (ORDER BY period) AS prev_price
            FROM price_records
            {lookback}
        )
        SELECT
            prev_period,
            prev_price,
            period AS current_period,
            price  AS current_price,
            units,
            (price - prev_price) AS abs_change,
            CASE WHEN prev_price = 0 THEN NULL
                 ELSE ((price - prev_price) / prev_price * 100.0) END AS pct_change
        FROM ordered
        WHERE prev_price IS NOT NULL
          {current_clause}
        ORDER BY {order_clause}
        LIMIT :k
    """

    with session_scope() as session:
        rows = [dict(r._mapping) for r in session.execute(text(sql), params)]

    return [
        _build_change(
            calculation_type="month_over_month",
            series_id=series_id,
            units=row["units"] or "",
            prev_period=row["prev_period"],
            prev_value=float(row["prev_price"]),
            curr_period=row["current_period"],
            curr_value=float(row["current_price"]),
        )
        for row in rows
    ]


def endpoint_change(series_id: str, period: TimePeriod) -> PriceChange | None:
    """First → last observation change within the period (end-to-end)."""
    params: dict = {"sid": series_id}
    clause = _period_clause(period, params)
    sql = f"""
        WITH ranked AS (
            SELECT period, price, units,
                   ROW_NUMBER() OVER (ORDER BY period ASC)  AS rn_asc,
                   ROW_NUMBER() OVER (ORDER BY period DESC) AS rn_desc
            FROM price_records
            WHERE series_id = :sid {clause}
        )
        SELECT
            (SELECT period FROM ranked WHERE rn_asc = 1)  AS first_period,
            (SELECT price  FROM ranked WHERE rn_asc = 1)  AS first_price,
            (SELECT period FROM ranked WHERE rn_desc = 1) AS last_period,
            (SELECT price  FROM ranked WHERE rn_desc = 1) AS last_price,
            (SELECT units  FROM ranked WHERE rn_asc = 1)  AS units
    """
    with session_scope() as session:
        row = session.execute(text(sql), params).mappings().one_or_none()
    if not row or row["first_price"] is None or row["last_price"] is None:
        return None
    if row["first_period"] == row["last_period"]:
        return None
    return _build_change(
        calculation_type="period_end_to_end",
        series_id=series_id,
        units=row["units"] or "",
        prev_period=row["first_period"],
        prev_value=float(row["first_price"]),
        curr_period=row["last_period"],
        curr_value=float(row["last_price"]),
    )
