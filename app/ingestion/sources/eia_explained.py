"""EIA "Electricity explained" pages (authoritative, timeless explainers).

These pages directly address what drives electricity prices (fuel costs,
generation mix, demand, weather, regulation), giving the knowledge base durable
explanatory coverage that is not tied to a single publication date.
"""
from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from app.ingestion.html import html_to_segments
from app.ingestion.sources.base import USER_AGENT, SourceDocument

SOURCE_NAME = "U.S. Energy Information Administration"
SOURCE_TYPE = "EIA Explainer"

BASE = "https://www.eia.gov/electricity/"
EXPLAINED_PAGES: list[tuple[str, str]] = [
    (BASE + "prices-and-factors-affecting-prices.php",
     "Electricity explained: Prices and factors affecting prices"),
    (BASE + "how-electricity-is-generated.php",
     "Electricity explained: How electricity is generated"),
    (BASE + "electricity-in-the-us.php",
     "Electricity explained: Electricity in the United States"),
    (BASE + "electricity-in-the-us-generation-capacity-and-sales.php",
     "Electricity explained: U.S. generation capacity and sales"),
    (BASE + "delivery-to-consumers.php",
     "Electricity explained: Delivery to consumers"),
    (BASE + "use-of-electricity.php",
     "Electricity explained: Use of electricity"),
    (BASE + "energy-storage-for-electricity-generation.php",
     "Electricity explained: Energy storage for electricity generation"),
]


def fetch_page(url: str, title: str | None = None) -> SourceDocument | None:
    resp = httpx.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    body = soup.select_one("div.article-content") or soup.select_one("div#page-content")
    if body is None:
        return None

    segments = html_to_segments(str(body))
    if not segments:
        return None

    doc_title = title
    if not doc_title:
        h1 = soup.find("h1")
        doc_title = h1.get_text(" ", strip=True) if h1 else url

    return SourceDocument(
        title=doc_title,
        source_name=SOURCE_NAME,
        source_type=SOURCE_TYPE,
        source_url=url,
        segments=segments,
        published_date=None,
    )
