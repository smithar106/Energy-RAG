"""CLI: ingest a text/HTML document into the knowledge base.

Usage:
    python -m scripts.ingest --title "..." --source wikipedia \
        --url https://... path/to/file.txt [--start 2022-01-01 --end 2023-01-01]
"""
from __future__ import annotations

import argparse
from datetime import date

from app.db.base import init_db
from app.ingestion.pipeline import ingest_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document")
    parser.add_argument("path")
    parser.add_argument("--title", required=True)
    parser.add_argument("--source", default="manual")
    parser.add_argument("--url", default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    init_db()

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    with open(args.path, encoding="utf-8") as f:
        text = f.read()

    doc_id, n_chunks = ingest_document(
        text=text,
        title=args.title,
        source=args.source,
        source_url=args.url,
        start_date=start,
        end_date=end,
    )
    print(f"ingested document id={doc_id} chunks={n_chunks}")


if __name__ == "__main__":
    main()
