"""Audit knowledge-base coverage by year / energy type / source / market layer.

    python -m scripts.audit_knowledge_coverage
    python -m scripts.audit_knowledge_coverage --api-url https://<host> --start 1995 --end 2026
"""
from __future__ import annotations

import argparse

from sqlalchemy import func, select

from app.db.base import session_scope
from app.db.models import Chunk, Document


def _audit_local(start: int, end: int) -> None:
    with session_scope() as session:
        rows = session.execute(
            select(Chunk.start_year, func.count()).group_by(Chunk.start_year)
        ).all()
        by_year = {y: c for y, c in rows if y is not None}

        print(f"{'year':>6}  {'EIA docs':>9}  {'wiki docs':>10}  {'elec':>5}  {'gas':>5}  {'retail':>7}  {'chunks':>7}")
        for year in range(start, end + 1):
            eia_docs = session.scalar(
                select(func.count(func.distinct(Chunk.document_id))).where(
                    Chunk.source_name == "U.S. Energy Information Administration",
                    Chunk.start_year == year,
                )
            ) or 0
            wiki_docs = session.scalar(
                select(func.count(func.distinct(Chunk.document_id))).where(
                    Chunk.source_name == "Wikipedia", Chunk.start_year == year
                )
            ) or 0
            elec = session.scalar(
                select(func.count()).where(Chunk.energy_type == "electricity", Chunk.start_year == year)
            ) or 0
            gas = session.scalar(
                select(func.count()).where(Chunk.energy_type == "natural_gas", Chunk.start_year == year)
            ) or 0
            retail = session.scalar(
                select(func.count()).where(Chunk.market_layer == "retail_price", Chunk.start_year == year)
            ) or 0
            chunks = by_year.get(year, 0)
            if chunks or eia_docs or wiki_docs:
                print(f"{year:>6}  {eia_docs:>9}  {wiki_docs:>10}  {elec:>5}  {gas:>5}  {retail:>7}  {chunks:>7}")

        # Totals by energy type / market layer.
        print("\nenergy_type distribution:")
        for et, cnt in session.execute(select(Chunk.energy_type, func.count()).group_by(Chunk.energy_type)).all():
            print(f"  {et or '(unlabelled)':>14}: {cnt}")
        print("market_layer distribution:")
        for ml, cnt in session.execute(select(Chunk.market_layer, func.count()).group_by(Chunk.market_layer)).all():
            print(f"  {ml or '(unlabelled)':>14}: {cnt}")
        n_docs = session.scalar(select(func.count()).select_from(Document)) or 0
        print(f"\ndocuments: {n_docs}")


def _audit_remote(api_url: str, start: int, end: int) -> None:
    import httpx
    data = httpx.get(api_url.rstrip("/") + "/admin/coverage", timeout=60).json()
    print(f"documents: {data['documents']}  chunks: {data['chunks']}")
    print(f"{'year':>6}  {'eia':>5}  {'wiki':>5}  {'elec':>5}  {'gas':>5}  {'retail':>7}  {'chunks':>7}")
    years = data.get("years", {})
    for year in range(start, end + 1):
        y = years.get(str(year))
        if y:
            print(f"{year:>6}  {y.get('eia',0):>5}  {y.get('wiki',0):>5}  "
                  f"{y.get('electricity',0):>5}  {y.get('natural_gas',0):>5}  "
                  f"{y.get('retail_price',0):>7}  {y.get('chunks',0):>7}")
    print("\nenergy_type:", {k or '(unlabelled)': v for k, v in data.get("energy_types", {}).items()})
    print("market_layer:", {k or '(unlabelled)': v for k, v in data.get("market_layers", {}).items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit KB coverage")
    parser.add_argument("--start", type=int, default=1995)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--api-url", default=None)
    args = parser.parse_args()

    if args.api_url:
        _audit_remote(args.api_url, args.start, args.end)
        return

    from app.db.base import init_db
    init_db()
    _audit_local(args.start, args.end)


if __name__ == "__main__":
    main()
