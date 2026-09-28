"""Backfill energy-domain + temporal metadata for existing chunks.

Populates the new columns (energy_type, market_layer, geography, sector,
mentioned_years, event_start_date, event_end_date) from already-stored chunk
text, without re-fetching sources.

    python -m scripts.backfill_metadata
"""
from __future__ import annotations

from app.db.base import init_db, session_scope
from app.db.models import Chunk
from app.ingestion.domain import classify_domain
from app.ingestion.metadata import extract_event_window, extract_mentioned_years


def main() -> None:
    init_db()
    with session_scope() as session:
        chunks = session.query(Chunk).all()
        updated = 0
        for chunk in chunks:
            blob = f"{chunk.document_title or ''}. {chunk.text or ''}"
            domain = classify_domain(blob)
            event_start, event_end = extract_event_window(
                chunk.text or "",
                title=chunk.document_title or "",
                published_date=chunk.published_date,
            )
            chunk.energy_type = domain.energy_type
            chunk.market_layer = domain.market_layer
            chunk.geography = domain.geography
            chunk.sector = domain.sector
            chunk.mentioned_years = extract_mentioned_years(blob)
            chunk.event_start_date = event_start
            chunk.event_end_date = event_end
            updated += 1
        session.commit()
    print(f"backfilled metadata for {updated} chunks")


if __name__ == "__main__":
    main()
