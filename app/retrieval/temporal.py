"""Temporal filtering.

Extracts a time period from a natural-language question and applies it to both
the SQL path (period filters) and the vector path (chunk event-window filters).

Handles explicit ranges ("between 2021 and 2023", "2014-2016"), single years,
named months, and a small set of well-known event aliases (e.g. COVID-19).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

_RANGE_RE = re.compile(
    r"(1[89]\d{2}|20\d{2})\s*(?:-|–|—|to|through|and)\s*(1[89]\d{2}|20\d{2})",
    re.I,
)
_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

# Well-known event aliases → event window (start_year, end_year).
_PERIOD_ALIASES: dict[str, tuple[int, int]] = {
    "covid-19": (2020, 2021),
    "covid": (2020, 2021),
    "coronavirus": (2020, 2021),
    "great recession": (2007, 2009),
    "financial crisis": (2007, 2009),
    "global financial crisis": (2007, 2009),
}


@dataclass
class TimePeriod:
    start: date | None = None
    end: date | None = None

    @property
    def is_bounded(self) -> bool:
        return self.start is not None or self.end is not None

    def overlaps(self, start: date | None, end: date | None) -> bool:
        if not self.is_bounded:
            return True
        if start is None and end is None:
            return True
        lo = start or date.min
        hi = end or date.max
        return (self.start is None or hi >= self.start) and (
            self.end is None or lo <= self.end
        )


def parse_time_period(question: str) -> TimePeriod:
    """Best-effort deterministic period extraction from a question."""
    q = (question or "").lower()

    for alias, (start_year, end_year) in _PERIOD_ALIASES.items():
        if alias in q:
            return TimePeriod(start=date(start_year, 1, 1), end=date(end_year, 12, 31))

    range_match = _RANGE_RE.search(q)
    if range_match:
        y1, y2 = int(range_match.group(1)), int(range_match.group(2))
        lo, hi = sorted((y1, y2))
        return TimePeriod(start=date(lo, 1, 1), end=date(hi, 12, 31))

    year_match = _YEAR_RE.search(q)
    if not year_match:
        return TimePeriod()
    year = int(year_match.group(1))

    month: int | None = None
    for name, m in _MONTHS.items():
        if name in q:
            month = m
            break

    if month:
        start = date(year, month, 1)
        end = (date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)) - timedelta(days=1)
        return TimePeriod(start=start, end=end)

    return TimePeriod(start=date(year, 1, 1), end=date(year, 12, 31))


def year_range_filter(period: TimePeriod) -> tuple[str, dict]:
    """Return a SQLAlchemy filter fragment for ``price_records.period``."""
    if not period.is_bounded:
        return "TRUE", {}
    if period.start and period.end:
        return "period BETWEEN :t_start AND :t_end", {
            "t_start": period.start,
            "t_end": period.end,
        }
    if period.start:
        return "period >= :t_start", {"t_start": period.start}
    return "period <= :t_end", {"t_end": period.end}
