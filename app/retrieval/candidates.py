"""Shared candidate SQL fragments for vector + lexical retrieval."""
from __future__ import annotations

from app.retrieval.temporal import TimePeriod

# Columns every candidate record must carry (self-describing for rerank/gate).
CHUNK_FIELDS = """
    c.id, c.document_id, c.chunk_index, c.section,
    c.source_name, c.source_type, c.document_title, c.source_url,
    c.published_date, c.start_year, c.end_year,
    c.event_start_date, c.event_end_date, c.mentioned_years,
    c.energy_type, c.market_layer, c.geography, c.sector,
    c.text
"""

# Energy types acceptable as *candidates* for a given intent energy type.
# (The evidence gate is stricter than candidate retrieval.)
COMPATIBLE_ENERGY = {
    "electricity": {"electricity", "multi_energy", "natural_gas", "coal"},
    "natural_gas": {"natural_gas", "multi_energy", "electricity"},
    "petroleum": {"petroleum", "multi_energy"},
    "gasoline": {"gasoline", "petroleum", "multi_energy"},
    "coal": {"coal", "multi_energy", "electricity"},
}


def period_clause(period: TimePeriod, params: dict, alias: str = "c") -> str:
    clause = ""
    if period.start:
        params["t_start_year"] = period.start.year
        clause += f" AND ({alias}.end_year IS NULL OR {alias}.end_year >= :t_start_year)"
    if period.end:
        params["t_end_year"] = period.end.year
        clause += f" AND ({alias}.start_year IS NULL OR {alias}.start_year <= :t_end_year)"
    return clause


def domain_clause(
    energy_type: str | None,
    geography: str | None,
    params: dict,
    alias: str = "c",
) -> str:
    """Permissive candidate-level filter: keep compatible or unlabelled chunks."""
    clause = ""
    if energy_type:
        allowed = COMPATIBLE_ENERGY.get(energy_type)
        if allowed:
            literals = ", ".join(f"'{a}'" for a in sorted(allowed))
            clause += f" AND ({alias}.energy_type IS NULL OR {alias}.energy_type IN ({literals}))"
    if geography:
        params["geo"] = geography
        clause += f" AND ({alias}.geography IS NULL OR {alias}.geography = :geo)"
    return clause
