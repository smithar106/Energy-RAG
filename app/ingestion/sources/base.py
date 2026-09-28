"""Shared types for source ingesters."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.ingestion.html import Segment

USER_AGENT = "Energy-RAG/0.1 (+https://github.com/smithar106/Energy-RAG)"


@dataclass
class SourceDocument:
    """A fetched, cleaned source document ready to chunk + embed."""

    title: str
    source_name: str
    source_type: str
    source_url: str
    segments: list[Segment] = field(default_factory=list)
    published_date: date | None = None
    revision_id: str | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)
