"""Wikipedia ingestion via the MediaWiki API (secondary source).

Uses ``action=query`` for existence/redirect resolution and ``action=parse`` to
retrieve the real article HTML (which preserves section headings) plus the
revision id. Only articles that actually exist are ingested.
"""
from __future__ import annotations

from urllib.parse import quote

import httpx

from app.ingestion.html import html_to_segments
from app.ingestion.sources.base import USER_AGENT, SourceDocument

API_URL = "https://en.wikipedia.org/w/api.php"
SOURCE_NAME = "Wikipedia"
SOURCE_TYPE = "Wikipedia"

# Curated seed list of energy-price-history articles. Titles are resolved
# (redirects followed); missing titles are skipped.
SEED_TITLES: list[str] = [
    "2021–2023 global energy crisis",
    "2000s energy crisis",
    "1970s energy crisis",
    "2020 Russia–Saudi Arabia oil price war",
    "2010s oil glut",
    "Energy crisis",
    "Electricity pricing",
    "Natural gas prices",
    "Price of oil",
    "Petroleum industry",
    "Shale gas",
    "Hydraulic fracturing in the United States",
]


def _api_get(params: dict) -> dict:
    merged = {**params, "format": "json", "formatversion": "2"}
    resp = httpx.get(
        API_URL,
        params=merged,
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()


def resolve_title(title: str) -> str | None:
    """Return the canonical title if the article exists, else None."""
    data = _api_get({"action": "query", "redirects": "1", "titles": title})
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    if page.get("missing"):
        return None
    return page.get("title")


def fetch_article(title: str) -> SourceDocument | None:
    """Fetch a real Wikipedia article as a :class:`SourceDocument`."""
    resolved = resolve_title(title)
    if resolved is None:
        return None

    data = _api_get(
        {
            "action": "parse",
            "page": resolved,
            "prop": "text|revid",
            "redirects": "1",
            "disabletoc": "1",
        }
    )
    parse = data.get("parse")
    if not parse:
        return None

    raw_text = parse.get("text")
    html = raw_text.get("*") if isinstance(raw_text, dict) else raw_text
    if not html:
        return None

    canonical_title = parse.get("title", resolved)
    revision_id = parse.get("revid")
    url = "https://en.wikipedia.org/wiki/" + quote(canonical_title.replace(" ", "_"))

    segments = html_to_segments(html)
    if not segments:
        return None

    return SourceDocument(
        title=canonical_title,
        source_name=SOURCE_NAME,
        source_type=SOURCE_TYPE,
        source_url=url,
        segments=segments,
        published_date=None,  # Wikipedia pages have no single publication date
        revision_id=str(revision_id) if revision_id else None,
    )
