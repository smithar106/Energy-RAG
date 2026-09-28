"""Ingestion orchestration.

Document → clean → chunk → local embedding → pgvector.

Price series → structured rows in ``price_records``.
"""
from __future__ import annotations

from datetime import date

from app.db.base import session_scope
from app.db.models import Chunk, Document, PriceRecord
from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_text
from app.ingestion.eia import EIAPoint
from app.providers.embeddings import get_embedding_provider


def ingest_document(
    *,
    text: str,
    title: str,
    source: str,
    source_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[int, int]:
    """Clean, chunk, embed, and store a document + its chunks."""
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)

    embedder = get_embedding_provider()
    embeddings = embedder.embed_documents([c.text for c in chunks])

    with session_scope() as session:
        doc = Document(
            title=title,
            source=source,
            source_url=source_url,
        )
        session.add(doc)
        session.flush()  # assign doc.id

        for chunk, vector in zip(chunks, embeddings):
            session.add(
                Chunk(
                    document_id=doc.id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    embedding=vector,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        session.flush()
        return doc.id, len(chunks)


def ingest_price_points(
    *,
    series_id: str,
    points: list[EIAPoint],
    region: str | None = None,
    fuel: str | None = None,
    source: str = "EIA",
) -> int:
    """Store verified EIA price points into the structured table."""
    from datetime import datetime

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
