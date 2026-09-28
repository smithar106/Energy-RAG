"""Retrieval service — the inspectable two-stage RAG retrieval pipeline.

    Stage 1  retrieval query → local embedding → pgvector candidates (30–50)
    Stage 2  hybrid ranking (semantic + temporal + authority)
             → evidence threshold + document diversity → accepted top-N

Returns a :class:`RetrievalResult` carrying the accepted evidence and the full
candidate list (with component scores and accept/reject state) for the trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.agent.prompts import build_retrieval_query
from app.config import get_settings
from app.providers.embeddings import get_embedding_provider
from app.providers.llm import ChatMessage, get_llm_provider
from app.retrieval.ranking import (
    DEFAULT_WEIGHTS,
    MIN_EVIDENCE_SCORE,
    RANKING_FORMULA,
    RELATIVE_FLOOR,
    rank_chunks,
    select_evidence,
)
from app.retrieval.temporal import TimePeriod
from app.retrieval.vector import vector_search


@dataclass
class RetrievalResult:
    retrieval_query: str
    period: TimePeriod
    query_embedding: list[float]
    candidate_count: int
    ranked: list[dict] = field(default_factory=list)       # accepted evidence
    ranked_all: list[dict] = field(default_factory=list)   # all candidates + accept state
    top_n: int = 0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    formula: str = RANKING_FORMULA


def make_retrieval_query(question: str) -> str:
    """Use DeepSeek to turn a question into a focused semantic query."""
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
    ranked_all = rank_chunks(candidates, period=period, top_n=None)
    accepted, _ = select_evidence(ranked_all, top_n)

    accepted_ids = {c["id"] for c in accepted}
    best = ranked_all[0]["final_score"] if ranked_all else 0.0
    for c in ranked_all:
        if c["id"] in accepted_ids:
            c["accepted"] = True
            c["rejection_reason"] = None
        else:
            c["accepted"] = False
            if c["final_score"] < MIN_EVIDENCE_SCORE:
                c["rejection_reason"] = "below evidence threshold"
            elif c["final_score"] < best * RELATIVE_FLOOR:
                c["rejection_reason"] = "below relative floor"
            else:
                c["rejection_reason"] = "document cap (diversity)"

    return RetrievalResult(
        retrieval_query=retrieval_query,
        period=period,
        query_embedding=query_vector,
        candidate_count=len(candidates),
        ranked=accepted,
        ranked_all=ranked_all,
        top_n=top_n,
    )
