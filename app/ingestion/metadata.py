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
from datetime import date

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
