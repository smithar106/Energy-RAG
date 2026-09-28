"""Seed the database with demo structured price data + a sample document.

This lets the whole pipeline run end-to-end without any external calls, so you
can verify the hybrid retrieval without waiting on the EIA API.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from app.db.base import init_db
from app.db.models import PriceRecord
from app.db.base import session_scope
from app.ingestion.pipeline import ingest_document

SAMPLE_DOC = {
    "title": "Historical context: U.S. retail electricity prices",
    "source": "seed",
    "text": (
        "U.S. retail electricity prices rose sharply in 2022 as natural gas prices "
        "spiked following the Russian invasion of Ukraine. Natural gas is a key "
        "fuel for marginal electricity generation, so higher gas prices flowed "
        "through to wholesale and retail power markets. In 2023 prices moderated "
        "as gas prices declined and weather was milder. Electricity prices vary "
        "widely by region, with the Northeast and California historically above "
        "the national average."
    ),
    "start_date": date(2022, 1, 1),
    "end_date": date(2023, 12, 31),
}


def seed_prices() -> None:
    rng = random.Random(42)
    series = "ELEC.PRICE.US-ALL.M"  # average retail electricity price, monthly
    start = date(2021, 1, 1)
    records = []
    for i in range(36):  # 3 years monthly
        period = date(start.year + (start.month - 1 + i) // 12, (start.month - 1 + i) % 12 + 1, 1)
        base = 13.0 + (period.year - 2021) * 0.6
        # 2022 spike + noise
        spike = 3.0 if period.year == 2022 else 0.0
        price = round(base + spike + rng.uniform(-0.4, 0.4), 2)
        records.append(
            PriceRecord(
                series_id=series,
                period=period,
                price=price,
                units="cents/kWh",
                region="US-ALL",
                fuel="all",
                source="seed",
            )
        )
    with session_scope() as session:
        session.add_all(records)
    print(f"seeded {len(records)} price records")


def main() -> None:
    init_db()
    seed_prices()
    doc_id, n_chunks = ingest_document(
        text=SAMPLE_DOC["text"],
        title=SAMPLE_DOC["title"],
        source=SAMPLE_DOC["source"],
        start_date=SAMPLE_DOC["start_date"],
        end_date=SAMPLE_DOC["end_date"],
    )
    print(f"seeded document id={doc_id} chunks={n_chunks}")


if __name__ == "__main__":
    main()
