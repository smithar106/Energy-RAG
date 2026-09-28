"""Ingestion orchestration.

    real source → clean → section-aware chunk → local embedding → pgvector

No text is generated or summarized here: every stored chunk is verbatim source
content from the fetched document.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import select

from app.db.base import session_scope
from app.db.models import Chunk, Document, PriceRecord
from app.ingestion.chunker import chunk_segments
from app.ingestion.domain import classify_domain
from app.ingestion.html import Segment, text_to_segments
from app.ingestion.metadata import (
    extract_event_window,
    extract_mentioned_years,
    infer_year_range,
)
from app.ingestion.sources.base import SourceDocument
from app.providers.embeddings import get_embedding_provider


def compute_document_metadata(
    title: str,
    full_text: str,
    published_date: date | None,
) -> dict:
    """Document-level metadata (domain + temporal), consistent across chunks."""
    domain = classify_domain(title + ". " + full_text)
    start_year, end_year = infer_year_range(full_text, published_date=published_date)
    event_start, event_end = extract_event_window(
        full_text, title=title, published_date=published_date
    )
    return {
        "energy_type": domain.energy_type,
        "market_layer": domain.market_layer,
        "geography": domain.geography,
        "sector": domain.sector,
        "start_year": start_year,
        "end_year": end_year,
        "event_start_date": event_start,
        "event_end_date": event_end,
        "mentioned_years": extract_mentioned_years(full_text),
    }


def store_source_document(
    doc: SourceDocument,
    *,
    force: bool = False,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[int, int]:
    """Chunk, embed, and store a source document. Returns (document_id, chunks).

    De-duplicates by ``source_url``. With ``force=True`` an existing document is
    deleted and replaced. Explicit ``start_year``/``end_year`` override the
    content-inferred event window.
    """
    chunks = chunk_segments(doc.segments)
    if not chunks:
        raise ValueError(f"no chunks produced for {doc.source_url}")

    if start_year is None and end_year is None:
        start_year, end_year = infer_year_range(
            doc.full_text, published_date=doc.published_date
        )

    event_start, event_end = extract_event_window(
        doc.full_text, title=doc.title, published_date=doc.published_date
    )
    domain = classify_domain(doc.title + ". " + doc.full_text)
    mentioned_years = extract_mentioned_years(doc.full_text)

    embedder = get_embedding_provider()
    embeddings = embedder.embed_documents([c.text for c in chunks])

    with session_scope() as session:
        existing = session.scalar(
            select(Document).where(Document.source_url == doc.source_url)
        )
        if existing is not None:
            if not force:
                return existing.id, 0
            session.delete(existing)
            session.flush()

        document = Document(
            title=doc.title,
            source_name=doc.source_name,
            source_type=doc.source_type,
            source_url=doc.source_url,
            published_date=doc.published_date,
            revision_id=doc.revision_id,
        )
        session.add(document)
        session.flush()

        for chunk, vector in zip(chunks, embeddings):
            session.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=chunk.index,
                    section=chunk.section,
                    source_name=doc.source_name,
                    source_type=doc.source_type,
                    document_title=doc.title,
                    source_url=doc.source_url,
                    published_date=doc.published_date,
                    start_year=start_year,
                    end_year=end_year,
                    event_start_date=event_start,
                    event_end_date=event_end,
                    mentioned_years=mentioned_years,
                    energy_type=domain.energy_type,
                    market_layer=domain.market_layer,
                    geography=domain.geography,
                    sector=domain.sector,
                    text=chunk.text,
                    embedding=vector,
                )
            )
        session.flush()
        return document.id, len(chunks)


def store_raw_document(
    *,
    title: str,
    source_name: str,
    source_type: str,
    source_url: str,
    text: str,
    published_date: date | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    section: str | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Store a caller-supplied real document (API path)."""
    doc = SourceDocument(
        title=title,
        source_name=source_name,
        source_type=source_type,
        source_url=source_url,
        segments=text_to_segments(text, default_heading=section),
        published_date=published_date,
    )
    return store_source_document(
        doc, force=force, start_year=start_year, end_year=end_year
    )


def ingest_price_points(
    *,
    series_id: str,
    points,
    region: str | None = None,
    fuel: str | None = None,
    source: str = "EIA",
) -> int:
    """Store verified EIA price points into the structured table."""
    with session_scope() as session:
        for p in points:
            period = datetime.strptime(p.period, "%Y-%m").date()
            session.add(
                PriceRecord(
                    series_id=series_id,
                    period=period,
                    price=p.value,
                    units=p.units,
                    region=region,
                    fuel=fuel,
                    source=source,
                )
            )
        return len(points)
