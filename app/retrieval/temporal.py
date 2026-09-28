"""Temporal filtering.

Extracts a time period from a natural-language question and applies it to both
the SQL path (period filters) and the vector path (chunk ``start_date`` /
``end_date`` overlap).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# Ordered, case-insensitive: most specific first.
_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"(\d{4})-(\d{2})"), 1),          # 2020-03
    (re.compile(r"(\d{4})"), 2),                  # 2020
    (re.compile(r"20\d\d"), 2),                   # 20xx
]

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


@dataclass
class TimePeriod:
    start: date | None = None
    end: date | None = None

    @property
    def is_bounded(self) -> bool:
        return self.start is not None or self.end is not None

    def overlaps(self, start: date | None, end: date | None) -> bool:
        """Does a chunk [start, end] overlap this period?"""
        if not self.is_bounded:
            return True
        if start is None and end is None:
            return True  # chunk has no temporal metadata → keep (can't filter)
        lo = start or date.min
        hi = end or date.max
        return (self.start is None or hi >= self.start) and (
            self.end is None or lo <= self.end
        )


def parse_time_period(question: str) -> TimePeriod:
    """Best-effort deterministic period extraction from a question."""
    q = question.lower()
    year_match = re.search(r"(19|20)\d{2}", q)
    if not year_match:
        return TimePeriod()

    year = int(year_match.group(0))

    # A named month near the year narrows the period to that month.
    month: int | None = None
    for name, m in _MONTHS.items():
        if name in q:
            month = m
            break

    if month:
        start = date(year, month, 1)
        end = date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)
        from datetime import timedelta

        end = end - timedelta(days=1)
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
