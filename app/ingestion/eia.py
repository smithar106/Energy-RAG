"""U.S. EIA API client — the authoritative source for structured price data.

Uses the EIA v2 backward-compatibility endpoint::

    https://api.eia.gov/v2/seriesid/{V1_SERIES_ID}

which is how the retired v1 series ids (e.g. ``ELEC.PRICE.US-ALL.M``) are
reached on the v2 API. The value field and its units differ by series
(``price`` / ``price-units`` for retail electricity, ``value`` elsewhere), so
the client auto-detects them from the response.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import get_settings

_BASE = "https://api.eia.gov/v2"

# Preferred field names, in order.
_VALUE_FIELDS = ("price", "value", "data")


@dataclass
class EIAPoint:
    period: str
    value: float
    units: str


class EIAClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.eia_api_key:
            raise RuntimeError("EIA_API_KEY is not set")
        self.api_key = settings.eia_api_key

    def fetch_series(
        self,
        series_id: str,
        *,
        length: int = 5000,
        start: str | None = None,
        end: str | None = None,
    ) -> list[EIAPoint]:
        """Fetch a series and return (period, value, units) points, ascending."""
        params: dict[str, Any] = {
            "api_key": self.api_key,
            "length": str(length),
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        url = f"{_BASE}/seriesid/{series_id}"
        resp = httpx.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
        payload = resp.json()

        response = payload.get("response", {})
        data = response.get("data", [])
        units = response.get("unit") or ""

        points: list[EIAPoint] = []
        for row in data:
            value_field = _detect_value_field(row)
            if value_field is None:
                continue
            value = row[value_field]
            if value is None:
                continue
            row_units = units or _detect_units(row, value_field)
            points.append(
                EIAPoint(period=row["period"], value=float(value), units=row_units)
            )

        points.sort(key=lambda p: p.period)
        return points


def _detect_value_field(row: dict) -> str | None:
    for field in _VALUE_FIELDS:
        if isinstance(row.get(field), (int, float)):
            return field
    # Fallback: first numeric field that is not a date/period column.
    for k, v in row.items():
        if k in ("period", "year", "month"):
            continue
        if isinstance(v, (int, float)):
            return k
    return None


def _detect_units(row: dict, value_field: str) -> str:
    for candidate in (f"{value_field}-units", f"{value_field}_units", "units"):
        if isinstance(row.get(candidate), str):
            return row[candidate]
    return ""
