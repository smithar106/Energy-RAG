"""Vector similarity search over pgvector.

Retrieval path:

    retrieval query → local embedding → pgvector cosine similarity → top-k chunks
"""
from __future__ import annotations

from sqlalchemy import text

from app.db.base import session_scope
from app.providers.embeddings import get_embedding_provider
from app.retrieval.temporal import TimePeriod


def vector_search(
    retrieval_query: str,
    *,
    period: TimePeriod,
    top_k: int,
) -> list[dict]:
    """Return top-k chunks by cosine similarity, filtered by time period."""
    embedder = get_embedding_provider()
    query_vector = embedder.embed_query(retrieval_query)

    params: dict = {
        "qvec": query_vector,
        "top_k": top_k,
    }
    period_clause = ""
    if period.start:
        params["t_start"] = period.start
        period_clause += " AND (c.end_date IS NULL OR c.end_date >= :t_start)"
    if period.end:
        params["t_end"] = period.end
        period_clause += " AND (c.start_date IS NULL OR c.start_date <= :t_end)"

    sql = f"""
        SELECT
            c.id,
            c.text,
            d.title,
            d.source,
            d.source_url,
            c.start_date,
            c.end_date,
            1 - (c.embedding <=> :qvec) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
          {period_clause}
        ORDER BY c.embedding <=> :qvec
        LIMIT :top_k
    """

    with session_scope() as session:
        rows = [
            dict(row._mapping)
            for row in session.execute(text(sql), params)
        ]

    for r in rows:
        r["similarity"] = float(r["similarity"])
        if r.get("start_date"):
            r["start_date"] = r["start_date"].isoformat()
        if r.get("end_date"):
            r["end_date"] = r["end_date"].isoformat()
    return rows
