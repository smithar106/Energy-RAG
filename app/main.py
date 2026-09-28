"""FastAPI application — the HTTP entrypoint.

The embedding model is loaded ONCE at startup (lifespan) and reused. The full
pipeline is::

    question
      -> DeepSeek agent (tool orchestration)
      -> { SQL numbers } + { pgvector evidence }
      -> DeepSeek grounded synthesis
      -> grounding validator (quantitative + explanatory + citation integrity)
      -> cited response (+ optional RAG trace)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from sqlalchemy import func, select

from app.agent.orchestrator import AgentOrchestrator
from app.config import get_settings
from app.db.base import init_db, reset_rag_tables, session_scope
from app.db.models import Chunk, Document, PriceRecord
from app.ingestion.eia import EIAClient
from app.ingestion.pipeline import ingest_price_points, store_raw_document, store_source_document
from app.ingestion.sources import eia_articles, wikipedia
from app.providers.embeddings import get_embedding_provider
from app.retrieval.service import make_retrieval_query, retrieve
from app.retrieval.temporal import TimePeriod, parse_time_period
from app.schemas.api import (
    AskRequest,
    AskResponse,
    ChunkTrace,
    IngestEIAResponse,
    IngestRequest,
    IngestResponse,
    IngestSourceResponse,
    RAGTrace,
    ResetResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.synthesis.citations import build_citations
from app.synthesis.grounding import validate_grounding
from app.synthesis.synthesizer import Synthesizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.database_url:
        init_db()
    get_embedding_provider()  # warm once
    yield


app = FastAPI(title="Energy-RAG", version="0.2.0", lifespan=lifespan)


# ── Web UI ─────────────────────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "llm_provider": "deepseek",
        "llm_model": settings.deepseek_model,
    }


# ── Ask ────────────────────────────────────────────────────────────────────
def _build_trace(question: str, agent_result, settings) -> RAGTrace | None:
    t = agent_result.retrieval_trace
    if not t:
        return None
    top_n = int(t.get("top_n") or 0)
    chunks = []
    for rank, c in enumerate(t.get("chunks", [])):
        chunks.append(
            ChunkTrace(
                rank=rank,
                chunk_id=c.get("id"),
                document_id=c.get("document_id"),
                document_title=c.get("document_title") or "",
                section=c.get("section"),
                source_name=c.get("source_name") or "",
                source_type=c.get("source_type") or "",
                source_url=c.get("source_url") or "",
                published_date=c.get("published_date"),
                start_year=c.get("start_year"),
                end_year=c.get("end_year"),
                semantic_similarity=c.get("semantic_similarity", 0.0),
                temporal_score=c.get("temporal_score", 0.0),
                temporal_reason=c.get("temporal_reason"),
                authority_score=c.get("authority_score", 0.0),
                final_score=c.get("final_score", 0.0),
                text=c.get("text", ""),
                passed_to_llm=rank < top_n,
            )
        )
    return RAGTrace(
        question=question,
        requested_period=t.get("requested_period", {}),
        retrieval_query=t.get("retrieval_query"),
        embedding_model=settings.embedding_model,
        embedding_dim=int(t.get("embedding_dim") or 0),
        query_embedding=[float(x) for x in t.get("query_embedding", [])],
        candidate_count=int(t.get("candidate_count") or 0),
        top_n=top_n,
        ranking_weights=t.get("weights", {}),
        ranking_formula=t.get("formula", ""),
        chunks=chunks,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    settings = get_settings()

    agent_result = AgentOrchestrator().run(question)
    answer = Synthesizer().synthesize(question, agent_result.sql_results, agent_result.evidence)
    grounding = validate_grounding(answer, agent_result.sql_results, agent_result.evidence)
    citations = build_citations(agent_result.evidence)

    trace = _build_trace(question, agent_result, settings) if req.include_trace else None
    return AskResponse(
        answer=answer,
        citations=citations,
        sql_results=agent_result.sql_results,
        retrieved_chunks=len(agent_result.evidence),
        grounding=grounding,
        trace=trace,
    )


@app.post("/debug/retrieval", response_model=RetrieveResponse)
def debug_retrieval(req: RetrieveRequest) -> RetrieveResponse:
    settings = get_settings()
    if req.time_period_start or req.time_period_end:
        period = TimePeriod(start=req.time_period_start, end=req.time_period_end)
    else:
        period = parse_time_period(req.query)

    query = make_retrieval_query(req.query) if req.generate_query else req.query
    result = retrieve(
        query, period, candidate_k=req.candidate_k, top_n=req.top_n
    )

    class _Agent:
        retrieval_trace = {
            "retrieval_query": result.retrieval_query,
            "requested_period": {
                "start": result.period.start.isoformat() if result.period.start else None,
                "end": result.period.end.isoformat() if result.period.end else None,
            },
            "query_embedding": result.query_embedding,
            "embedding_dim": len(result.query_embedding),
            "candidate_count": result.candidate_count,
            "top_n": result.top_n,
            "weights": result.weights,
            "formula": result.formula,
            "chunks": result.ranked_all,
        }

    trace = _build_trace(req.query, _Agent(), settings)
    trace.generated_retrieval_query = query if req.generate_query else None
    return RetrieveResponse(trace=trace)


# ── Ingestion ──────────────────────────────────────────────────────────────
@app.post("/ingest/eia", response_model=IngestEIAResponse)
def ingest_eia(
    series_id: str,
    length: int = 5000,
    region: str | None = None,
    fuel: str | None = None,
) -> IngestEIAResponse:
    points = EIAClient().fetch_series(series_id, length=length)
    if not points:
        raise HTTPException(status_code=404, detail="no data for series")
    ingest_price_points(series_id=series_id, points=points, region=region, fuel=fuel)
    return IngestEIAResponse(
        series_id=series_id, records=len(points), start=points[0].period, end=points[-1].period
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    doc_id, n_chunks = store_raw_document(
        title=req.title,
        source_name=req.source_name,
        source_type=req.source_type,
        source_url=req.source_url,
        text=req.text,
        published_date=req.published_date,
        start_year=req.start_year,
        end_year=req.end_year,
        section=req.section,
        force=req.force,
    )
    return IngestResponse(
        document_id=doc_id,
        chunks=n_chunks,
        embedding_model=get_settings().embedding_model,
        source_url=req.source_url,
    )


@app.post("/admin/ingest/eia", response_model=list[IngestSourceResponse])
def admin_ingest_eia(
    offset: int = 0, limit: int = 1, force: bool = False
) -> list[IngestSourceResponse]:
    items = eia_articles.list_recent_articles(limit=offset + limit)[offset : offset + limit]
    results: list[IngestSourceResponse] = []
    for item in items:
        doc = eia_articles.fetch_article(
            item.url, title=item.title, published_date=item.published_date
        )
        if doc is None:
            continue
        doc_id, n = store_source_document(doc, force=force)
        results.append(
            IngestSourceResponse(
                title=doc.title,
                source_name=doc.source_name,
                source_url=doc.source_url,
                document_id=doc_id,
                chunks=n,
                published_date=doc.published_date,
            )
        )
    return results


@app.post("/admin/ingest/wikipedia", response_model=IngestSourceResponse)
def admin_ingest_wikipedia(title: str, force: bool = False) -> IngestSourceResponse:
    doc = wikipedia.fetch_article(title)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"wikipedia article not found: {title}")
    doc_id, n = store_source_document(doc, force=force)
    return IngestSourceResponse(
        title=doc.title,
        source_name=doc.source_name,
        source_url=doc.source_url,
        document_id=doc_id,
        chunks=n,
    )


@app.post("/admin/reset-rag", response_model=ResetResponse)
def admin_reset_rag() -> ResetResponse:
    reset_rag_tables()
    return ResetResponse(dropped=["chunks", "documents"], recreated=["documents", "chunks"])


@app.get("/admin/stats")
def admin_stats() -> dict:
    with session_scope() as session:
        n_docs = session.scalar(select(func.count()).select_from(Document)) or 0
        n_chunks = session.scalar(select(func.count()).select_from(Chunk)) or 0
        by_source = session.execute(
            select(Document.source_name, func.count()).group_by(Document.source_name)
        ).all()
        by_chunks = session.execute(
            select(Chunk.source_name, func.count()).group_by(Chunk.source_name)
        ).all()
        pub_min, pub_max = session.execute(
            select(func.min(Document.published_date), func.max(Document.published_date))
        ).one()
        temporal_chunks = session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.start_year.isnot(None))
        ) or 0
        n_prices = session.scalar(select(func.count()).select_from(PriceRecord)) or 0

    return {
        "documents": n_docs,
        "chunks": n_chunks,
        "documents_by_source": {name: cnt for name, cnt in by_source},
        "chunks_by_source": {name: cnt for name, cnt in by_chunks},
        "earliest_publication_date": pub_min.isoformat() if pub_min else None,
        "latest_publication_date": pub_max.isoformat() if pub_max else None,
        "chunks_with_temporal_metadata": temporal_chunks,
        "price_records": n_prices,
    }
