"""Source metadata inference.

Temporal metadata is stored in two separate fields on purpose:

- ``published_date`` — when the *source* was published.
- ``start_year`` / ``end_year`` — the *event window* the source discusses,
  inferred from the source content. Publication year ≠ event year.

The event window is inferred as a **contiguous cluster of years around the most
frequently mentioned year**, which keeps a topical article tight (e.g. a
2021–2023 crisis article → 2021–2023) instead of a broad min/max that would
make every source temporally match everything.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import date, timedelta

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

# Hard ceiling on how wide an inferred event window may grow.
_MAX_SPAN_YEARS = 30


def infer_year_range(
    text: str,
    *,
    published_date: date | None = None,
) -> tuple[int | None, int | None]:
    """Infer (start_year, end_year) as a contiguous cluster of mentioned years."""
    years = [int(m) for m in _YEAR_RE.findall(text or "")]
    if not years:
        return (None, None)

    counts = Counter(years)
    center = max(counts, key=lambda y: (counts[y], y))
    present = set(years)

    start = end = center
    while (start - 1) in present and (end - start) < _MAX_SPAN_YEARS:
        start -= 1
    while (end + 1) in present and (end - start) < _MAX_SPAN_YEARS:
        end += 1

    if published_date is not None:
        # An event cannot post-date its reporting (allow +1 for outlooks).
        end = min(end, published_date.year + 1)
        start = min(start, end)

    return (start, end)


_MONTH_NUMS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RANGE_RE = re.compile(
    r"\b([A-Za-z]{3,9})\s*(?:[-–—]|to|through|and)\s*([A-Za-z]{3,9})\s*(?:of\s+)?(\d{4})\b"
)
_MONTH_YEAR_RE = re.compile(r"\b([A-Za-z]{3,9})\s+(\d{4})\b")
_YEAR_RANGE_RE = re.compile(
    r"\b(1[89]\d{2}|20\d{2})\s*(?:[-–—]|to|through|and)\s*(1[89]\d{2}|20\d{2})\b"
)


def _month_end(month: int, year: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)


def extract_mentioned_years(text: str) -> list[int]:
    return sorted({int(m) for m in _YEAR_RE.findall(text or "")})


def extract_event_window(
    text: str,
    *,
    title: str = "",
    published_date: date | None = None,
) -> tuple[date | None, date | None]:
    """Infer (event_start_date, event_end_date) with month-level precision.

    Title and lead often carry the strongest event-period signal for EIA
    articles (e.g. "May-June 2017"). Falls back to a year cluster.
    """
    blob = f"{title or ''}. {text or ''}"

    m = _MONTH_RANGE_RE.search(blob)
    if m:
        m1 = _MONTH_NUMS.get(m.group(1).lower())
        m2 = _MONTH_NUMS.get(m.group(2).lower())
        year = int(m.group(3))
        if m1 and m2 and m1 <= m2:
            return date(year, m1, 1), _month_end(m2, year) - timedelta(days=1)

    m = _MONTH_YEAR_RE.search(blob)
    if m:
        month = _MONTH_NUMS.get(m.group(1).lower())
        year = int(m.group(2))
        if month:
            d = date(year, month, 1)
            return d, _month_end(month, year) - timedelta(days=1)

    m = _YEAR_RANGE_RE.search(blob)
    if m:
        y1, y2 = sorted((int(m.group(1)), int(m.group(2))))
        return date(y1, 1, 1), date(y2, 12, 31)

    start_year, end_year = infer_year_range(blob, published_date=published_date)
    if start_year is None:
        return None, None
    return date(start_year, 1, 1), date(end_year, 12, 31)
