"""CLI: fetch an EIA series and store it into the structured table.

Usage:
    python -m scripts.fetch_eia --series ELEC.PRICE.US-ALL.M --length 5000
"""
from __future__ import annotations

import argparse

from app.db.base import init_db
from app.ingestion.eia import EIAClient
from app.ingestion.pipeline import ingest_price_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch + store an EIA series")
    parser.add_argument("--series", required=True)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--region", default=None)
    parser.add_argument("--fuel", default=None)
    args = parser.parse_args()

    init_db()
    client = EIAClient()
    points = client.fetch_series(args.series, length=args.length)
    n = ingest_price_points(
        series_id=args.series,
        points=points,
        region=args.region,
        fuel=args.fuel,
    )
    print(f"stored {n} records for {args.series} ({points[0].period}..{points[-1].period})")


if __name__ == "__main__":
    main()
