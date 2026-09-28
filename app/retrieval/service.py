"""Retrieval service — the inspectable RAG retrieval pipeline.

    retrieval query → local embedding → pgvector candidates
                    → hybrid ranking (semantic + temporal + authority) → top-N

Returns a :class:`RetrievalResult` carrying everything needed for the RAG trace:
the query, the embedding, the candidate count, and every ranked chunk with its
component scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.agent.prompts import build_retrieval_query
from app.config import get_settings
from app.providers.embeddings import get_embedding_provider
from app.providers.llm import ChatMessage, get_llm_provider
from app.retrieval.ranking import (
    DEFAULT_WEIGHTS,
    RANKING_FORMULA,
    rank_chunks,
)
from app.retrieval.temporal import TimePeriod
from app.retrieval.vector import vector_search


@dataclass
class RetrievalResult:
    retrieval_query: str
    period: TimePeriod
    query_embedding: list[float]
    candidate_count: int
    ranked: list[dict] = field(default_factory=list)       # top-N passed to synthesis
    ranked_all: list[dict] = field(default_factory=list)   # full ranked candidate list
    top_n: int = 0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    formula: str = RANKING_FORMULA


def make_retrieval_query(question: str) -> str:
    """Use DeepSeek to turn a question into a focused retrieval query."""
    llm = get_llm_provider()
    response = llm.chat(
        [ChatMessage(role="user", content=build_retrieval_query(question))],
        temperature=0.0,
    )
    query = (response.get("content") or "").strip().strip('"')
    return query or question


def retrieve(
    retrieval_query: str,
    period: TimePeriod,
    *,
    candidate_k: int | None = None,
    top_n: int | None = None,
) -> RetrievalResult:
    settings = get_settings()
    candidate_k = candidate_k or settings.retrieval_top_k
    top_n = top_n or settings.rerank_top_n

    embedder = get_embedding_provider()
    query_vector = embedder.embed_query(retrieval_query)

    candidates = vector_search(
        retrieval_query,
        period=period,
        top_k=candidate_k,
        query_vector=query_vector,
    )
    ranked_all = rank_chunks(candidates, period=period, top_n=candidate_k)
    ranked = ranked_all[:top_n]

    return RetrievalResult(
        retrieval_query=retrieval_query,
        period=period,
        query_embedding=query_vector,
        candidate_count=len(candidates),
        ranked=ranked,
        ranked_all=ranked_all,
        top_n=top_n,
    )
