"""Permanently remove synthetic/placeholder RAG content.

Drops and recreates ONLY ``documents`` / ``chunks``. ``price_records`` (real EIA
data) is never touched.

    python -m scripts.reset_rag_content
    python -m scripts.reset_rag_content --api-url https://<host>
"""
from __future__ import annotations

import argparse

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset RAG tables (keep price_records)")
    parser.add_argument("--api-url", default=None, help="Remote API base URL")
    args = parser.parse_args()

    if args.api_url:
        r = httpx.post(args.api_url.rstrip("/") + "/admin/reset-rag", timeout=120.0)
        r.raise_for_status()
        print(r.json())
    else:
        from app.db.base import reset_rag_tables

        reset_rag_tables()
        print("dropped + recreated: documents, chunks (price_records untouched)")


if __name__ == "__main__":
    main()
