"""Ingest real Wikipedia articles into pgvector.

Local mode (direct DB):
    python -m scripts.ingest_wikipedia                 # curated seed list
    python -m scripts.ingest_wikipedia --title "Shale gas"

Remote mode (deployed API):
    python -m scripts.ingest_wikipedia --api-url https://<host>
"""
from __future__ import annotations

import argparse

import httpx

from app.ingestion.sources import wikipedia


def _run_local(titles: list[str], force: bool) -> None:
    from app.db.base import init_db
    from app.ingestion.pipeline import store_source_document

    init_db()
    for title in titles:
        doc = wikipedia.fetch_article(title)
        if doc is None:
            print(f"  SKIP (not found): {title}")
            continue
        doc_id, n = store_source_document(doc, force=force)
        print(f"  doc={doc_id} chunks={n} {doc.title}")


def _run_remote(api_url: str, titles: list[str], force: bool) -> None:
    base = api_url.rstrip("/")
    for title in titles:
        r = httpx.post(
            f"{base}/admin/ingest/wikipedia",
            params={"title": title, "force": force},
            timeout=180.0,
        )
        if r.status_code == 404:
            print(f"  SKIP (not found): {title}")
            continue
        r.raise_for_status()
        row = r.json()
        print(f"  doc={row['document_id']} chunks={row['chunks']} {row['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia articles")
    parser.add_argument("--title", action="append", default=None,
                        help="Specific title(s); default = curated seed list")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--api-url", default=None, help="Remote API base URL")
    args = parser.parse_args()

    titles = args.title or wikipedia.SEED_TITLES
    if args.api_url:
        _run_remote(args.api_url, titles, args.force)
    else:
        _run_local(titles, args.force)


if __name__ == "__main__":
    main()
