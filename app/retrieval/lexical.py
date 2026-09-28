"""Lexical search via PostgreSQL full-text (candidate retrieval stage 1B).

Combined with pgvector this makes retrieval genuinely hybrid: exact terms such
as "2017", "June", "electricity", "retail", "price" are matched lexically even
when semantic similarity is weak.
"""
from __future__ import annotations

import re

from sqlalchemy import text

from app.db.base import session_scope
from app.retrieval.candidates import CHUNK_FIELDS, domain_clause, period_clause
from app.retrieval.temporal import TimePeriod

_TERM_RE = re.compile(r"[a-z0-9]+")


def _to_tsquery(query: str) -> str:
    """Build a disjunctive tsquery of meaningful tokens (deduplicated)."""
    seen: set[str] = set()
    terms: list[str] = []
    for token in _TERM_RE.findall((query or "").lower()):
        if token in seen:
            continue
        if len(token) <= 2 and not token.isdigit():
            continue
        seen.add(token)
        terms.append(token)
    return " | ".join(terms) if terms else ""


def lexical_search(
    query: str,
    *,
    period: TimePeriod,
    top_k: int,
    energy_type: str | None = None,
    geography: str | None = None,
) -> list[dict]:
    """Return candidate chunks matching the query lexically (ts_rank)."""
    tsquery = _to_tsquery(query)
    if not tsquery:
        return []

    params: dict = {"q": tsquery, "top_k": top_k}
    filters = period_clause(period, params) + domain_clause(energy_type, geography, params)

    sql = f"""
        SELECT {CHUNK_FIELDS},
               ts_rank(
                   to_tsvector('english',
                       coalesce(c.document_title,'') || ' ' ||
                       coalesce(c.section,'') || ' ' || c.text),
                   to_tsquery('english', :q)
               ) AS lexical_score
        FROM chunks c
        WHERE to_tsvector('english',
                 coalesce(c.document_title,'') || ' ' ||
                 coalesce(c.section,'') || ' ' || c.text) @@ to_tsquery('english', :q)
          {filters}
        ORDER BY ts_rank(
                   to_tsvector('english',
                       coalesce(c.document_title,'') || ' ' ||
                       coalesce(c.section,'') || ' ' || c.text),
                   to_tsquery('english', :q)) DESC
        LIMIT :top_k
    """

    with session_scope() as session:
        rows = [dict(r._mapping) for r in session.execute(text(sql), params)]

    for r in rows:
        r["lexical_score"] = float(r["lexical_score"])
    return rows
