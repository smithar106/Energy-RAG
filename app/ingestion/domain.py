"""Energy-domain classification.

Assigns each chunk structured labels — ``energy_type``, ``market_layer``,
``geography``, ``sector`` — so retrieval can reject off-domain candidates (e.g.
a gasoline-formulation article surfacing for a retail-electricity question)
instead of relying on semantic similarity alone.
"""
from __future__ import annotations

from dataclasses import dataclass

# Keyword categories (lowercased substrings). Ordered for deterministic ties.
ENERGY_TYPE_KEYWORDS: dict[str, list[str]] = {
    "electricity": [
        "electricity", "electric power", "power plant", "power plants",
        "kilowatt", "megawatt", "gigawatt", "electric grid", "power grid",
        "retail electricity", "power prices", "power generation",
    ],
    "natural_gas": [
        "natural gas", "henry hub", "lng", "gas-fired", "gas fired",
        "natural-gas", "gas prices", "gas price",
    ],
    "petroleum": [
        "crude oil", "crude", "oil price", "oil prices", "petroleum",
        "oil glut", "oil production", "oil market", "oil shipments",
    ],
    "gasoline": [
        "gasoline", "motor gasoline", "motor fuel", "diesel", "distillate",
        "gas station", "at the pump", "crack spread",
    ],
    "coal": ["coal"],
}

MARKET_LAYER_KEYWORDS: dict[str, list[str]] = {
    "retail_price": [
        "retail price", "retail electricity", "residential price",
        "cents per kilowatt", "cents/kwh", "customer price",
        "price for customers", "price to consumers", "residential customers",
    ],
    "wholesale_price": ["wholesale", "spot price", "day-ahead", "power price"],
    "generation": ["generation", "power plant", "generator", "generated", "capacity"],
    "fuel_cost": ["fuel cost", "fuel price", "fuel prices", "fuel costs", "fuel"],
    "transmission": ["transmission"],
    "distribution": ["distribution"],
    "demand": ["demand", "load", "consumption", "electricity use", "peak load"],
    "supply": ["supply", "production", "output", "exports", "imports"],
    "policy": ["policy", "regulation", "subsidy", "tax credit", "mandate", "standard"],
    "infrastructure": ["infrastructure", "pipeline", "grid", "storage", "refinery", "terminal"],
}

GEOGRAPHY_KEYWORDS: dict[str, list[str]] = {
    "United States": ["united states", "u.s.", "u.s", "america", "american", " in the us "],
    "international": ["europe", "european", "asia", "china", "russia", "saudi",
                      "nigeria", "canada", "mexico", "global", "world"],
}

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "residential": ["residential", "household", "home"],
    "commercial": ["commercial"],
    "industrial": ["industrial"],
    "generation": ["generation", "power plant", "generator"],
    "all": ["all sectors"],
}


@dataclass
class DomainMeta:
    energy_type: str | None
    market_layer: str | None
    geography: str | None
    sector: str | None

    def as_dict(self) -> dict:
        return {
            "energy_type": self.energy_type,
            "market_layer": self.market_layer,
            "geography": self.geography,
            "sector": self.sector,
        }


def _top(text: str, table: dict[str, list[str]]) -> str | None:
    t = text.lower()
    best: str | None = None
    best_score = 0
    for label, keywords in table.items():
        score = sum(t.count(k) for k in keywords)
        if score > best_score:
            best_score = score
            best = label
    return best if best_score > 0 else None


def classify_energy_type(text: str) -> str | None:
    t = text.lower()
    scores = {k: sum(t.count(w) for w in ws) for k, ws in ENERGY_TYPE_KEYWORDS.items()}
    present = [k for k, v in scores.items() if v > 0]
    if not present:
        return None
    if len(present) >= 2:
        return "multi_energy"
    return present[0]


def classify_domain(text: str) -> DomainMeta:
    return DomainMeta(
        energy_type=classify_energy_type(text),
        market_layer=_top(text, MARKET_LAYER_KEYWORDS),
        geography=_top(text, GEOGRAPHY_KEYWORDS),
        sector=_top(text, SECTOR_KEYWORDS),
    )
