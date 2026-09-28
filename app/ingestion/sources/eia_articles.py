"""EIA Today in Energy ingestion (primary source).

Discovers recent analysis articles from the EIA RSS feed, fetches the real
article HTML, extracts the title / canonical URL / publication date / body
(with headings), and returns a :class:`SourceDocument`. No generation or
summarization happens here — only extraction of the actual source text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from app.ingestion.html import Segment, clean_inline, html_to_segments
from app.ingestion.sources.base import USER_AGENT, SourceDocument

RSS_URL = "https://www.eia.gov/rss/todayinenergy.xml"
SOURCE_NAME = "U.S. Energy Information Administration"
SOURCE_TYPE = "EIA Analysis"

# Leading analysis-type / date lines and chart captions to drop.
_BOILERPLATE_RE = re.compile(
    r"^(in-brief analysis|today in energy|data source:|source:|note:|"
    r"[a-z]+ \d{1,2}, \d{4})\b",
    re.I,
)

# A valid EIA article URL carries a numeric id, e.g. detail.php?id=68204.
_ARTICLE_URL_RE = re.compile(r"detail\.php\?id=\d+", re.I)


def _valid_link(link: str) -> bool:
    return bool(link) and bool(_ARTICLE_URL_RE.search(link))


@dataclass
class EIAItem:
    title: str
    url: str
    published_date: date
    description: str


def _parse_pubdate(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        return None


def list_recent_articles(*, limit: int = 50) -> list[EIAItem]:
    resp = httpx.get(
        RSS_URL, headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    items: list[EIAItem] = []
    for node in root.findall(".//item")[:limit]:
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        published = _parse_pubdate(node.findtext("pubDate"))
        if title and _valid_link(link) and published:
            items.append(
                EIAItem(
                    title=title,
                    url=link,
                    published_date=published,
                    description=(node.findtext("description") or "").strip(),
                )
            )
    return items


def _strip_boilerplate(segments: list[Segment]) -> list[Segment]:
    cleaned: list[Segment] = []
    for seg in segments:
        paras = []
        for p in seg.text.split("\n\n"):
            p = clean_inline(p)
            if not p or _BOILERPLATE_RE.match(p):
                continue
            paras.append(p)
        if paras:
            cleaned.append(Segment(heading=seg.heading, text="\n\n".join(paras)))
    return cleaned


def fetch_article(
    url: str,
    *,
    title: str | None = None,
    published_date: date | None = None,
) -> SourceDocument | None:
    """Fetch + extract a single EIA analysis article."""
    resp = httpx.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    body = soup.select_one("div.tie-article") or soup.select_one("div#page-content")
    if body is None:
        return None

    # Prefer the (validated) feed URL; only accept a canonical tag if it is a
    # well-formed article URL.
    canonical_tag = soup.find("link", rel="canonical")
    canonical_candidate = canonical_tag.get("href") if canonical_tag else None
    source_url = (
        canonical_candidate
        if canonical_candidate and _valid_link(canonical_candidate)
        else url
    )
    if not _valid_link(source_url):
        return None

    doc_title = title
    if not doc_title:
        h1 = soup.find("h1")
        doc_title = h1.get_text(" ", strip=True) if h1 else source_url

    segments = _strip_boilerplate(html_to_segments(str(body)))
    if not segments:
        return None

    return SourceDocument(
        title=doc_title,
        source_name=SOURCE_NAME,
        source_type=SOURCE_TYPE,
        source_url=source_url,
        segments=segments,
        published_date=published_date,
    )
