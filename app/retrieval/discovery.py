"""Query-time authoritative source discovery.

When the local knowledge base yields insufficient evidence, this retrieves REAL
EIA documents for the event year(s) and ingests them, so a causal explanation is
still grounded in retrieved text + real URLs — never model memory. The evidence
gate is re-applied afterward; if nothing passes, the grounded refusal stands.
"""
from __future__ import annotations

from app.config import get_settings
from app.ingestion.sources import eia_articles
from app.ingestion.sources.base import SourceDocument
from app.retrieval.intent import RetrievalIntent

_ELECTRICITY_TERMS = (
    "electric", "power", "price", "generat", "utility", "retail", "coal",
    "natural gas", "demand", "weather", "wholesale", "grid", "transmission",
)


def _title_relevant(title: str, intent: RetrievalIntent) -> bool:
    t = (title or "").lower()
    energy = intent.energy_type
    if energy == "electricity":
        return any(k in t for k in _ELECTRICITY_TERMS)
    if energy == "natural_gas":
        return any(k in t for k in ("natural gas", "gas", "henry hub", "lng", "price"))
    if energy == "petroleum":
        return any(k in t for k in ("crude", "oil", "petroleum", "price", "gasoline"))
    # Unknown energy type → prefer price/cost/market articles.
    return any(k in t for k in ("price", "cost", "market", "supply", "demand"))


def _event_years(intent: RetrievalIntent) -> list[int]:
    if intent.event_start is None:
        return []
    year = intent.event_start.year
    if intent.is_causal:
        return sorted({year - 1, year, year + 1})
    return [year]


def discover_eia_documents(
    question: str,
    intent: RetrievalIntent,
    *,
    max_docs: int | None = None,
) -> list[SourceDocument]:
    """Fetch real EIA Today in Energy documents for the event year(s)."""
    settings = get_settings()
    max_docs = max_docs or settings.discovery_max_docs

    years = _event_years(intent)
    if not years:
        return []

    # Title-level prefilter (cheap), then fetch only the relevant articles.
    candidates: list[eia_articles.EIAItem] = []
    for year in years:
        for item in eia_articles.list_archive_by_year(year, limit=400):
            if _title_relevant(item.title, intent):
                candidates.append(item)

    candidates.sort(key=lambda i: i.published_date, reverse=True)

    docs: list[SourceDocument] = []
    seen: set[str] = set()
    for item in candidates:
        if item.url in seen:
            continue
        doc = eia_articles.fetch_article(
            item.url, title=item.title, published_date=item.published_date
        )
        if doc is None:
            continue
        seen.add(item.url)
        docs.append(doc)
        if len(docs) >= max_docs:
            break
    return docs
