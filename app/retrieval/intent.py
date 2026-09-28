"""Retrieval intent — structured dimensions parsed from question + SQL result.

The intent drives candidate filtering, query expansion, reranking, and the
evidence gate, so retrieval is anchored to the actual series and event.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from app.retrieval.series import series_meta
from app.retrieval.temporal import parse_time_period

_CAUSAL_RE = re.compile(
    r"\b(why|what caused|what drove|caused|reason|explain|led to|due to|"
    r"because|what made)\b",
    re.I,
)


@dataclass
class RetrievalIntent:
    energy_type: str | None
    market_layer: str | None
    geography: str | None
    sector: str | None
    event_start: date | None
    event_end: date | None
    is_causal: bool

    @property
    def context_start(self) -> date | None:
        if self.event_start is None:
            return None
        return date(self.event_start.year - 1, 1, 1)

    @property
    def context_end(self) -> date | None:
        if self.event_end is None:
            return None
        return date(self.event_end.year + 1, 12, 31)


def _metric_to_layer(metric: str | None) -> str | None:
    m = (metric or "").lower()
    if "retail" in m:
        return "retail_price"
    if "wholesale" in m:
        return "wholesale_price"
    if "fuel" in m or "natural gas price" in m:
        return "fuel_cost"
    return None


def _sector_label(sector: str | None) -> str | None:
    s = (sector or "").lower()
    if "all" in s:
        return "all"
    if "residential" in s:
        return "residential"
    if "commercial" in s:
        return "commercial"
    if "industrial" in s:
        return "industrial"
    return sector or None


def _series_from_sql(sql_results: list[dict]) -> str | None:
    for res in sql_results or []:
        pc = res.get("price_change")
        if pc and pc.get("series_id"):
            return pc["series_id"]
        changes = res.get("changes")
        if changes:
            return changes[0].get("series_id")
        rows = res.get("rows") or []
        if rows and rows[0].get("series_id"):
            return rows[0]["series_id"]
    return None


def _event_from_sql(sql_results: list[dict]) -> tuple[date | None, date | None]:
    """The specific event window (e.g. the largest change's May→June pair)."""
    for res in sql_results or []:
        changes = res.get("changes") or []
        if changes:
            c = changes[0]
            start = _parse_date(c.get("previous_period"))
            end = _parse_date(c.get("current_period"))
            if start and end:
                return start, end
    return None, None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_intent(question: str, sql_results: list[dict]) -> RetrievalIntent:
    series_id = _series_from_sql(sql_results)
    meta = series_meta(series_id)

    event_start, event_end = _event_from_sql(sql_results)
    if event_start is None or event_end is None:
        period = parse_time_period(question)
        event_start, event_end = period.start, period.end

    return RetrievalIntent(
        energy_type=meta.energy_type,
        market_layer=_metric_to_layer(meta.metric),
        geography=meta.geography,
        sector=_sector_label(meta.sector),
        event_start=event_start,
        event_end=event_end,
        is_causal=bool(_CAUSAL_RE.search(question or "")),
    )
