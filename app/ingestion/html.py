"""HTML / text extraction that preserves document structure.

Produces a list of :class:`Segment` (heading + paragraph-preserving text) so the
chunker can respect semantic boundaries instead of splitting on raw characters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

_SKIP_TAGS = {
    "script", "style", "nav", "footer", "header", "aside", "form",
    "table", "figure", "figcaption", "noscript", "iframe", "sup", "button",
}
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_TEXT_TAGS = ("p", "li", "blockquote", "dd")

# Wikipedia sections that are not article content.
_BOILERPLATE_HEADINGS = {
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "citations", "sources", "footnotes", "notes and references",
    "explanatory notes", "general sources", "works cited", "data sources",
}


@dataclass
class Segment:
    heading: str | None
    text: str  # paragraphs separated by blank lines


def clean_inline(text: str) -> str:
    """Normalize a single block of text: strip citation markers + whitespace."""
    text = re.sub(r"\[(?:edit|note|citation needed|clarification needed|when\?)\]",
                  " ", text, flags=re.I)
    text = re.sub(r"\[\d+\]", " ", text)          # [1], [23]
    text = re.sub(r"\[[a-z]{1,3}\]", " ", text)   # [a], [nb 1]
    text = re.sub(r"\[\s*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def html_to_segments(html: str, default_heading: str | None = None) -> list[Segment]:
    """Split article HTML into heading-scoped segments of paragraphs."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(_SKIP_TAGS)):
        tag.decompose()

    segments: list[Segment] = []
    current_heading = default_heading
    paragraphs: list[str] = []

    def flush() -> None:
        nonlocal paragraphs
        if paragraphs:
            segments.append(Segment(heading=current_heading, text="\n\n".join(paragraphs)))
            paragraphs = []

    for el in soup.find_all(list(_HEADING_TAGS) + list(_TEXT_TAGS)):
        if el.name in _HEADING_TAGS:
            flush()
            heading = clean_inline(el.get_text(" ", strip=True))
            if heading:
                current_heading = heading
        else:
            text = clean_inline(el.get_text(" ", strip=True))
            if len(text) > 2:
                paragraphs.append(text)
    flush()

    # Drop boilerplate sections.
    return [
        s for s in segments
        if not (s.heading and s.heading.strip().lower() in _BOILERPLATE_HEADINGS)
    ]


def text_to_segments(text: str, default_heading: str | None = None) -> list[Segment]:
    """Fallback for already-plain text: split into paragraph groups."""
    paragraphs = [clean_inline(p) for p in re.split(r"\n{1,}", text or "")]
    paragraphs = [p for p in paragraphs if len(p) > 2]
    if not paragraphs:
        return []
    return [Segment(heading=default_heading, text="\n\n".join(paragraphs))]
