"""Backfill energy-domain + temporal metadata for existing chunks.

Populates the new columns (energy_type, market_layer, geography, sector,
mentioned_years, event_start_date, event_end_date) from already-stored chunk
text, without re-fetching sources.

    python -m scripts.backfill_metadata
"""
from __future__ import annotations

from app.db.base import init_db, session_scope
from app.db.models import Document
from app.ingestion.pipeline import compute_document_metadata


def main() -> None:
    init_db()
    with session_scope() as session:
        docs = session.query(Document).all()
        updated = 0
        for doc in docs:
            full_text = doc.title + ". " + " ".join(ch.text for ch in doc.chunks)
            meta = compute_document_metadata(doc.title, full_text, doc.published_date)
            for chunk in doc.chunks:
                chunk.energy_type = meta["energy_type"]
                chunk.market_layer = meta["market_layer"]
                chunk.geography = meta["geography"]
                chunk.sector = meta["sector"]
                chunk.start_year = meta["start_year"]
                chunk.end_year = meta["end_year"]
                chunk.event_start_date = meta["event_start_date"]
                chunk.event_end_date = meta["event_end_date"]
                chunk.mentioned_years = meta["mentioned_years"]
                updated += 1
        session.commit()
    print(f"backfilled metadata for {updated} chunks")


if __name__ == "__main__":
    main()
