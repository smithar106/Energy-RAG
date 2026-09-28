"""Three-stage hybrid retrieval service.

    VECTOR CANDIDATES (pgvector)  +  LEXICAL CANDIDATES (tsvector)
              └────────── merge ──────────┘
                         │
                    RERANK (composite score)
                         │
                   EVIDENCE GATE (hard filters)
                         │
                   ACCEPTED EVIDENCE (diversity)

Returns an inspectable :class:`RetrievalResult` with every stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.providers.embeddings import get_embedding_provider
from app.providers.llm import ChatMessage, get_llm_provider
from app.retrieval.intent import RetrievalIntent, parse_intent
from app.retrieval.lexical import lexical_search
from app.retrieval.query_builder import build_retrieval_queries
from app.retrieval.ranking import (
    DEFAULT_WEIGHTS,
    RANKING_FORMULA,
    gate_chunks,
    rank_chunks,
)
from app.retrieval.temporal import TimePeriod
from app.retrieval.vector import vector_search


@dataclass
class RetrievalResult:
    intent: RetrievalIntent
    queries: list[str]
    query_embedding: list[float]
    vector_count: int
    lexical_count: int
    candidate_count: int
    ranked_all: list[dict] = field(default_factory=list)
    ranked: list[dict] = field(default_factory=list)       # accepted evidence
    top_n: int = 0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    formula: str = RANKING_FORMULA


def make_retrieval_query(question: str) -> str:
    """Use DeepSeek to turn a question into a focused semantic query."""
    from app.agent.prompts import build_retrieval_query

    llm = get_llm_provider()
    response = llm.chat(
        [ChatMessage(role="user", content=build_retrieval_query(question))],
        temperature=0.0,
    )
    query = (response.get("content") or "").strip().strip('"')
    return query or question


def retrieve(
    *,
    question: str,
    semantic_query: str,
    sql_results: list[dict],
    period: TimePeriod,
    candidate_k: int | None = None,
    top_n: int | None = None,
) -> RetrievalResult:
    settings = get_settings()
    candidate_k = candidate_k or settings.retrieval_top_k
    top_n = top_n or settings.rerank_top_n

    intent = parse_intent(question, sql_results)
    queries = build_retrieval_queries(semantic_query, intent)

    embedder = get_embedding_provider()
    primary_embedding = embedder.embed_query(queries[0] if queries else semantic_query)

    # Stage 1A: vector candidates.
    vector: dict[int, dict] = {}
    for q in queries:
        qvec = embedder.embed_query(q)
        for c in vector_search(
            q,
            period=period,
            top_k=candidate_k,
            query_vector=qvec,
            energy_type=intent.energy_type,
            geography=intent.geography,
        ):
            c["from_vector"] = True
            vector.setdefault(c["id"], c)

    # Stage 1B: lexical candidates.
    lexical: dict[int, dict] = {}
    for q in queries:
        for c in lexical_search(
            q,
            period=period,
            top_k=candidate_k,
            energy_type=intent.energy_type,
            geography=intent.geography,
        ):
            c["from_lexical"] = True
            c.setdefault("similarity", 0.0)
            lexical.setdefault(c["id"], c)

    # Merge (dedupe by chunk id).
    merged: dict[int, dict] = {}
    for cid, c in vector.items():
        merged[cid] = c
    for cid, c in lexical.items():
        if cid in merged:
            merged[cid]["from_lexical"] = True
        else:
            merged[cid] = c

    candidates = list(merged.values())
    query_text = " ".join(queries)

    # Stage 2: rerank (composite score).
    ranked_all = rank_chunks(candidates, intent=intent, query_text=query_text)

    # Stage 3: evidence gate (hard filters). `ranked` = all gate-passed chunks;
    # final diversity + causal filtering happen downstream in the ask flow.
    passed, rejected = gate_chunks(ranked_all, intent=intent)
    for c in passed:
        c["accepted"] = True
    for c in rejected:
        c["accepted"] = False

    combined = passed + rejected
    combined.sort(key=lambda r: r["final_score"], reverse=True)

    return RetrievalResult(
        intent=intent,
        queries=queries,
        query_embedding=primary_embedding,
        vector_count=len(vector),
        lexical_count=len(lexical),
        candidate_count=len(candidates),
        ranked_all=combined,
        ranked=passed,
        top_n=top_n,
    )
