"""Series metadata registry.

Maps EIA series ids to structured dimensions (energy type, metric, geography,
sector, unit) so retrieval queries and change records can carry them explicitly
and programmatically instead of relying on the model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeriesMeta:
    series_id: str
    energy_type: str | None
    metric: str | None
    geography: str | None
    sector: str | None
    unit: str | None


_REGISTRY: dict[str, SeriesMeta] = {
    "ELEC.PRICE.US-ALL.M": SeriesMeta(
        series_id="ELEC.PRICE.US-ALL.M",
        energy_type="electricity",
        metric="retail electricity price",
        geography="United States",
        sector="all sectors",
        unit="cents/kWh",
    ),
    "ELEC.PRICE.US-RES.M": SeriesMeta(
        series_id="ELEC.PRICE.US-RES.M",
        energy_type="electricity",
        metric="residential retail electricity price",
        geography="United States",
        sector="residential",
        unit="cents/kWh",
    ),
}

_SECTOR_NAMES = {
    "ALL": "all sectors",
    "RES": "residential",
    "COM": "commercial",
    "IND": "industrial",
    "TRA": "transportation",
}


def series_meta(series_id: str | None) -> SeriesMeta:
    if not series_id:
        return SeriesMeta("", None, None, None, None, None)
    if series_id in _REGISTRY:
        return _REGISTRY[series_id]

    # Heuristic fallback for the common EIA patterns.
    if series_id.startswith("ELEC.PRICE"):
        # e.g. ELEC.PRICE.US-ALL.M  -> geography=US, sector=ALL
        tail = series_id.split(".")[2] if series_id.count(".") >= 2 else ""
        geo, _, sector = tail.partition("-")
        geography = "United States" if geo.upper() in {"US", "USA"} else geo or None
        return SeriesMeta(
            series_id=series_id,
            energy_type="electricity",
            metric="retail electricity price",
            geography=geography,
            sector=_SECTOR_NAMES.get(sector.upper(), sector or None),
            unit="cents/kWh",
        )
    return SeriesMeta(series_id, None, None, None, None, None)
