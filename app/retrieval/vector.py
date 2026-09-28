"""Vector similarity search over pgvector.

Retrieval path:

    retrieval query → local embedding → pgvector cosine similarity → top-k chunks

Returns self-describing candidate records (provenance + temporal metadata) so
the ranking stage and the RAG trace can inspect every field without extra joins.
"""
from __future__ import annotations

from sqlalchemy import text

from app.db.base import session_scope
from app.providers.embeddings import get_embedding_provider
from app.retrieval.temporal import TimePeriod


def _to_vector_literal(values: list[float]) -> str:
    # pgvector's ``vector`` type is not a native psycopg type, so bind the
    # textual ``[1,2,3]`` form and CAST explicitly in SQL.
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def vector_search(
    retrieval_query: str,
    *,
    period: TimePeriod,
    top_k: int,
    query_vector: list[float] | None = None,
) -> list[dict]:
    """Return top-k candidate chunks by cosine similarity, filtered by period."""
    if query_vector is None:
        query_vector = get_embedding_provider().embed_query(retrieval_query)
    qvec = _to_vector_literal(query_vector)

    params: dict = {"qvec": qvec, "top_k": top_k}
    period_clause = ""
    if period.start:
        params["t_start_year"] = period.start.year
        period_clause += " AND (c.end_year IS NULL OR c.end_year >= :t_start_year)"
    if period.end:
        params["t_end_year"] = period.end.year
        period_clause += " AND (c.start_year IS NULL OR c.start_year <= :t_end_year)"

    sql = f"""
        SELECT
            c.id,
            c.document_id,
            c.chunk_index,
            c.section,
            c.source_name,
            c.source_type,
            c.document_title,
            c.source_url,
            c.published_date,
            c.start_year,
            c.end_year,
            c.text,
            1 - (c.embedding <=> CAST(:qvec AS vector)) AS similarity
        FROM chunks c
        WHERE c.embedding IS NOT NULL
          {period_clause}
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :top_k
    """

    with session_scope() as session:
        rows = [dict(r._mapping) for r in session.execute(text(sql), params)]

    for r in rows:
        r["similarity"] = float(r["similarity"])
    return rows
