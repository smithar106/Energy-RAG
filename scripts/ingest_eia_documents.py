"""Ingest real EIA Today in Energy / analysis articles into pgvector.

Local mode (direct DB):
    python -m scripts.ingest_eia_documents --limit 20

Remote mode (deployed API):
    python -m scripts.ingest_eia_documents --api-url https://<host> --limit 20
"""
from __future__ import annotations

import argparse

import httpx

from app.ingestion.sources import eia_articles


def _run_local(limit: int, force: bool) -> None:
    from app.db.base import init_db
    from app.ingestion.pipeline import store_source_document

    init_db()
    items = eia_articles.list_recent_articles(limit=limit)
    print(f"discovered {len(items)} EIA articles")
    for item in items:
        doc = eia_articles.fetch_article(
            item.url, title=item.title, published_date=item.published_date
        )
        if doc is None:
            print(f"  SKIP (no body): {item.url}")
            continue
        doc_id, n = store_source_document(doc, force=force)
        years = ""
        print(f"  doc={doc_id} chunks={n} pub={doc.published_date} {doc.title[:70]}")


def _run_remote(api_url: str, limit: int, force: bool) -> None:
    base = api_url.rstrip("/")
    for offset in range(limit):
        r = httpx.post(
            f"{base}/admin/ingest/eia",
            params={"offset": offset, "limit": 1, "force": force},
            timeout=180.0,
        )
        r.raise_for_status()
        for row in r.json():
            print(f"  doc={row['document_id']} chunks={row['chunks']} "
                  f"pub={row.get('published_date')} {row['title'][:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest EIA analysis articles")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--api-url", default=None, help="Remote API base URL")
    args = parser.parse_args()

    if args.api_url:
        _run_remote(args.api_url, args.limit, args.force)
    else:
        _run_local(args.limit, args.force)


if __name__ == "__main__":
    main()
