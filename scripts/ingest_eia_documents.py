"""Ingest real EIA electricity/analysis content into pgvector.

Sources:
  --recent     latest EIA Today in Energy (RSS)               [default]
  --archive    historical Today in Energy archive (dated)     ~all
  --explained  EIA "Electricity explained" pages (timeless)

Local mode (direct DB):
    python -m scripts.ingest_eia_documents --recent --archive --explained

Remote mode (deployed API):
    python -m scripts.ingest_eia_documents --api-url https://<host> \
        --recent --archive --explained
"""
from __future__ import annotations

import argparse

import httpx

from app.ingestion.sources import eia_articles, eia_explained


def _local_recent(limit: int, force: bool) -> None:
    from app.ingestion.pipeline import store_source_document

    items = eia_articles.list_recent_articles(limit=limit)
    print(f"recent: discovered {len(items)}")
    for item in items:
        doc = eia_articles.fetch_article(item.url, title=item.title, published_date=item.published_date)
        if doc is None:
            continue
        doc_id, n = store_source_document(doc, force=force)
        print(f"  doc={doc_id} chunks={n} pub={doc.published_date} {doc.title[:70]}")


def _local_archive(limit: int, force: bool) -> None:
    from app.ingestion.pipeline import store_source_document

    items = eia_articles.list_archive_articles(limit=limit)
    print(f"archive: discovered {len(items)}")
    for item in items:
        doc = eia_articles.fetch_article(item.url, title=item.title, published_date=item.published_date)
        if doc is None:
            continue
        doc_id, n = store_source_document(doc, force=force)
        print(f"  doc={doc_id} chunks={n} pub={doc.published_date} {doc.title[:70]}")


def _local_explained(force: bool) -> None:
    from app.ingestion.pipeline import store_source_document

    for url, title in eia_explained.EXPLAINED_PAGES:
        doc = eia_explained.fetch_page(url, title)
        if doc is None:
            print(f"  SKIP {url}")
            continue
        doc_id, n = store_source_document(doc, force=force)
        print(f"  doc={doc_id} chunks={n} {doc.title[:70]}")


def _remote(api_url: str, args) -> None:
    base = api_url.rstrip("/")
    if args.recent:
        for offset in range(args.limit):
            r = httpx.post(f"{base}/admin/ingest/eia",
                           params={"offset": offset, "limit": 1, "force": args.force}, timeout=180)
            r.raise_for_status()
            for row in r.json():
                print(f"  recent doc={row['document_id']} chunks={row['chunks']} {row['title'][:65]}")
    if args.archive:
        for offset in range(args.archive_limit):
            r = httpx.post(f"{base}/admin/ingest/eia-archive",
                           params={"offset": offset, "limit": 1, "force": args.force}, timeout=180)
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            for row in rows:
                print(f"  archive doc={row['document_id']} chunks={row['chunks']} pub={row.get('published_date')} {row['title'][:55]}")
    for year in args.archive_years:
        offset = 0
        while offset < args.archive_limit:
            r = httpx.post(f"{base}/admin/ingest/eia-archive-year",
                           params={"year": year, "offset": offset, "limit": 1, "force": args.force}, timeout=180)
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            for row in rows:
                print(f"  {year} doc={row['document_id']} chunks={row['chunks']} pub={row.get('published_date')} {row['title'][:50]}")
            offset += 1
    if args.explained:
        r = httpx.post(f"{base}/admin/ingest/eia-explained",
                       params={"force": args.force}, timeout=300)
        r.raise_for_status()
        for row in r.json():
            print(f"  explained doc={row['document_id']} chunks={row['chunks']} {row['title'][:60]}")


def _parse_years(value: str | None) -> list[int]:
    if not value:
        return []
    years = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        elif part:
            years.append(int(part))
    return sorted(set(years))


def _local_years(years: list[int], limit: int, force: bool) -> None:
    from app.ingestion.pipeline import store_source_document

    for year in years:
        items = eia_articles.list_archive_by_year(year, limit=limit)
        print(f"{year}: discovered {len(items)}")
        for item in items:
            doc = eia_articles.fetch_article(item.url, title=item.title, published_date=item.published_date)
            if doc is None:
                continue
            doc_id, n = store_source_document(doc, force=force)
            print(f"  doc={doc_id} chunks={n} pub={doc.published_date} {doc.title[:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest EIA electricity content")
    parser.add_argument("--recent", action="store_true", help="latest RSS articles")
    parser.add_argument("--archive", action="store_true", help="recent archive listing")
    parser.add_argument("--archive-years", default=None, help="historical years, e.g. '2011-2019' or '2017'")
    parser.add_argument("--explained", action="store_true", help="Electricity explained pages")
    parser.add_argument("--limit", type=int, default=20, help="recent article count")
    parser.add_argument("--archive-limit", type=int, default=400, help="per-year article cap")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--api-url", default=None)
    args = parser.parse_args()
    args.archive_years = _parse_years(args.archive_years)

    if not (args.recent or args.archive or args.archive_years or args.explained):
        args.recent = True

    if args.api_url:
        _remote(args.api_url, args)
        return

    from app.db.base import init_db
    init_db()
    if args.explained:
        _local_explained(args.force)
    if args.archive:
        _local_archive(args.archive_limit, args.force)
    if args.archive_years:
        _local_years(args.archive_years, args.archive_limit, args.force)
    if args.recent:
        _local_recent(args.limit, args.force)


if __name__ == "__main__":
    main()
