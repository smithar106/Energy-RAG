"""Verify the knowledge base and demonstrate real retrieval.

    python -m scripts.verify_db
    python -m scripts.verify_db --api-url https://<host>

Prints document/chunk counts by source, publication-date range, temporal
metadata coverage, and the top ranked chunks (with real URLs + component
scores) for several test queries.
"""
from __future__ import annotations

import argparse
import json

import httpx

TEST_QUERIES = [
    "Why did electricity prices increase between 2021 and 2023?",
    "Why did energy prices decline around 2015?",
    "What factors affected energy prices during COVID-19?",
]


def _print_stats_local() -> None:
    from sqlalchemy import func, select

    from app.db.base import session_scope
    from app.db.models import Chunk, Document, PriceRecord

    with session_scope() as s:
        print("documents:", s.scalar(select(func.count()).select_from(Document)))
        print("chunks:", s.scalar(select(func.count()).select_from(Chunk)))
        print("documents_by_source:",
              dict(s.execute(select(Document.source_name, func.count()).group_by(Document.source_name)).all()))
        print("chunks_by_source:",
              dict(s.execute(select(Chunk.source_name, func.count()).group_by(Chunk.source_name)).all()))
        print("publication_range:",
              s.execute(select(func.min(Document.published_date), func.max(Document.published_date))).one())
        print("chunks_with_temporal_metadata:",
              s.scalar(select(func.count()).select_from(Chunk).where(Chunk.start_year.isnot(None))))
        print("price_records:", s.scalar(select(func.count()).select_from(PriceRecord)))


def _run_query_local(query: str) -> None:
    from app.retrieval.service import retrieve
    from app.retrieval.temporal import parse_time_period

    result = retrieve(query, parse_time_period(query))
    print(f"\nQUERY: {query}")
    print(f"  retrieval_query={result.retrieval_query!r} period={result.period}")
    print(f"  candidates={result.candidate_count} top_n={result.top_n}")
    for rank, c in enumerate(result.ranked):
        print(f"  [{rank}] sem={c['semantic_similarity']} tmp={c['temporal_score']}"
              f"({c['temporal_reason']}) auth={c['authority_score']} final={c['final_score']}")
        print(f"      {c['source_name']} | {c['document_title']} | {c['source_url']}")
        print(f"      years={c['start_year']}-{c['end_year']} pub={c['published_date']}")
        print(f"      {c['text'][:200]}...")


def _run_query_remote(api_url: str, query: str) -> None:
    r = httpx.post(
        api_url.rstrip("/") + "/debug/retrieval",
        json={"query": query},
        timeout=180.0,
    )
    r.raise_for_status()
    trace = r.json()["trace"]
    print(f"\nQUERY: {query}")
    print(f"  retrieval_query={trace['retrieval_query']!r} period={trace['requested_period']}")
    print(f"  candidates={trace['candidate_count']} top_n={trace['top_n']}")
    for c in trace["chunks"][: trace["top_n"]]:
        print(f"  [{c['rank']}] sem={c['semantic_similarity']} tmp={c['temporal_score']}"
              f"({c['temporal_reason']}) auth={c['authority_score']} final={c['final_score']}")
        print(f"      {c['source_name']} | {c['document_title']} | {c['source_url']}")
        print(f"      years={c['start_year']}-{c['end_year']} pub={c['published_date']}")
        print(f"      {c['text'][:200]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify KB + run retrieval tests")
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    queries = args.query or TEST_QUERIES

    if args.api_url:
        stats = httpx.get(args.api_url.rstrip("/") + "/admin/stats", timeout=60.0).json()
        print(json.dumps(stats, indent=2))
        if not args.stats_only:
            for q in queries:
                _run_query_remote(args.api_url, q)
    else:
        _print_stats_local()
        if not args.stats_only:
            for q in queries:
                _run_query_local(q)


if __name__ == "__main__":
    main()
