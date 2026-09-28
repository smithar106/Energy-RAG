"""Vector similarity search over pgvector (candidate retrieval stage 1A)."""
from __future__ import annotations

from sqlalchemy import text

from app.db.base import session_scope
from app.providers.embeddings import get_embedding_provider
from app.retrieval.candidates import CHUNK_FIELDS, domain_clause, period_clause
from app.retrieval.temporal import TimePeriod


def _to_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def vector_search(
    retrieval_query: str,
    *,
    period: TimePeriod,
    top_k: int,
    query_vector: list[float] | None = None,
    energy_type: str | None = None,
    geography: str | None = None,
) -> list[dict]:
    """Return top-k candidate chunks by cosine similarity (+ metadata filters)."""
    if query_vector is None:
        query_vector = get_embedding_provider().embed_query(retrieval_query)
    qvec = _to_vector_literal(query_vector)

    params: dict = {"qvec": qvec, "top_k": top_k}
    filters = period_clause(period, params) + domain_clause(energy_type, geography, params)

    sql = f"""
        SELECT {CHUNK_FIELDS},
               1 - (c.embedding <=> CAST(:qvec AS vector)) AS similarity
        FROM chunks c
        WHERE c.embedding IS NOT NULL
          {filters}
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :top_k
    """

    with session_scope() as session:
        rows = [dict(r._mapping) for r in session.execute(text(sql), params)]

    for r in rows:
        r["similarity"] = float(r["similarity"])
    return rows
