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
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from sqlalchemy import func, select

from app.agent.orchestrator import AgentOrchestrator
from app.config import get_settings
from app.db.base import init_db, reset_rag_tables, session_scope
from app.db.models import Chunk, Document, PriceRecord
from app.ingestion.eia import EIAClient
from app.ingestion.pipeline import ingest_price_points, store_raw_document, store_source_document
from app.ingestion.sources import eia_articles, eia_explained, wikipedia
from app.providers.embeddings import get_embedding_provider
from app.retrieval.changes import largest_changes
from app.retrieval.intent import parse_intent
from app.retrieval.ranking import apply_diversity
from app.retrieval.service import make_retrieval_query, retrieve
from app.retrieval.sql import default_series_id
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
from app.synthesis.causal import filter_causally_useful
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
_CHANGE_HINT_RE = re.compile(
    r"\b(biggest|largest|most|greatest|steepest|highest|maximum)\b", re.I
)
_DECREASE_RE = re.compile(r"\b(decrease|drop|fall|decline|loss|lowest|minimum)\b", re.I)


def _supplement_largest_change(question: str, sql_results: list[dict]) -> None:
    """Deterministically compute the largest change when the question asks for it.

    Guarantees the LAG()-based observation pair is present even if the model
    forgot to call ``find_largest_change``, so the answer is grounded in SQL.
    """
    if not _CHANGE_HINT_RE.search(question):
        return
    for res in sql_results:
        if "changes" in res:
            return  # already computed by the agent

    direction = "decrease" if _DECREASE_RE.search(question) else "increase"
    period = parse_time_period(question)
    series_id = default_series_id()
    if not series_id or not period.is_bounded:
        return

    for metric in ("percent", "absolute"):
        changes = largest_changes(series_id, period, metric=metric, direction=direction)
        if changes:
            sql_results.append(
                {
                    "series_id": series_id,
                    "metric": metric,
                    "direction": direction,
                    "interpretation": f"largest month-over-month {metric} {direction}",
                    "changes": [c.to_dict() for c in changes],
                }
            )


def _build_trace(question: str, agent_result, settings) -> RAGTrace | None:
    t = agent_result.retrieval_trace
    if not t:
        return None
    top_n = int(t.get("top_n") or 0)
    chunks = []
    for rank, c in enumerate(t.get("chunks", [])):
        meta = c.get("temporal_meta") or {}
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
                event_start_date=c.get("event_start_date"),
                event_end_date=c.get("event_end_date"),
                energy_type=c.get("energy_type"),
                market_layer=c.get("market_layer"),
                geography=c.get("geography"),
                sector=c.get("sector"),
                semantic_similarity=c.get("semantic_similarity", 0.0),
                lexical_score=c.get("lexical_score", 0.0),
                temporal_score=c.get("temporal_score", 0.0),
                temporal_reason=c.get("temporal_reason"),
                temporal_coverage=meta.get("coverage"),
                temporal_specificity=meta.get("specificity"),
                chunk_span=meta.get("chunk_span"),
                domain_score=c.get("domain_score", 0.0),
                metric_score=c.get("metric_score", 0.0),
                geography_score=c.get("geography_score", 0.0),
                authority_score=c.get("authority_score", 0.0),
                final_score=c.get("final_score", 0.0),
                gate=c.get("gate"),
                failure_reason=c.get("failure_reason"),
                from_vector=bool(c.get("from_vector")),
                from_lexical=bool(c.get("from_lexical")),
                accepted=bool(c.get("accepted")),
                passed_to_llm=bool(c.get("accepted")),
                text=c.get("text", ""),
            )
        )
    return RAGTrace(
        question=question,
        intent=t.get("intent", {}),
        requested_period=t.get("requested_period", {}),
        queries=t.get("queries", []),
        embedding_model=settings.embedding_model,
        embedding_dim=int(t.get("embedding_dim") or 0),
        query_embedding=[float(x) for x in t.get("query_embedding", [])],
        vector_count=int(t.get("vector_count") or 0),
        lexical_count=int(t.get("lexical_count") or 0),
        candidate_count=int(t.get("candidate_count") or 0),
        accepted_count=int(t.get("accepted_count") or 0),
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
    _supplement_largest_change(question, agent_result.sql_results)

    # Evidence pipeline: gate-passed → causal-usefulness → diversity.
    intent = parse_intent(question, agent_result.sql_results)
    evidence = agent_result.evidence  # all gate-passed chunks
    if intent.is_causal:
        evidence = filter_causally_useful(question, evidence[:30])
    evidence = apply_diversity(evidence, settings.rerank_top_n)

    answer = Synthesizer().synthesize(question, agent_result.sql_results, evidence)
    grounding = validate_grounding(answer, agent_result.sql_results, evidence)
    citations = build_citations(evidence)

    trace = _build_trace(question, agent_result, settings) if req.include_trace else None
    if trace is not None:
        final_ids = {c["id"] for c in evidence}
        for c in trace.chunks:
            c.passed_to_llm = c.chunk_id in final_ids
    return AskResponse(
        answer=answer,
        citations=citations,
        sql_results=agent_result.sql_results,
        retrieved_chunks=len(evidence),
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

    semantic_query = make_retrieval_query(req.query) if req.generate_query else req.query
    result = retrieve(
        question=req.query,
        semantic_query=semantic_query,
        sql_results=[],
        period=period,
        candidate_k=req.candidate_k,
        top_n=req.top_n,
    )

    class _Agent:
        pass

    agent = _Agent()
    agent.retrieval_trace = {
        "intent": {
            "energy_type": result.intent.energy_type,
            "market_layer": result.intent.market_layer,
            "geography": result.intent.geography,
            "sector": result.intent.sector,
            "event_start": result.intent.event_start.isoformat() if result.intent.event_start else None,
            "event_end": result.intent.event_end.isoformat() if result.intent.event_end else None,
            "is_causal": result.intent.is_causal,
        },
        "requested_period": {
            "start": period.start.isoformat() if period.start else None,
            "end": period.end.isoformat() if period.end else None,
        },
        "queries": result.queries,
        "query_embedding": result.query_embedding,
        "embedding_dim": len(result.query_embedding),
        "vector_count": result.vector_count,
        "lexical_count": result.lexical_count,
        "candidate_count": result.candidate_count,
        "accepted_count": len(result.ranked),
        "top_n": result.top_n,
        "weights": result.weights,
        "formula": result.formula,
        "chunks": result.ranked_all,
    }

    trace = _build_trace(req.query, agent, settings)
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
        try:
            doc = eia_articles.fetch_article(
                item.url, title=item.title, published_date=item.published_date
            )
            if doc is None:
                continue
            doc_id, n = store_source_document(doc, force=force)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {item.url}: {exc}")
            continue
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


@app.post("/admin/ingest/eia-archive", response_model=list[IngestSourceResponse])
def admin_ingest_eia_archive(
    offset: int = 0, limit: int = 1, force: bool = False
) -> list[IngestSourceResponse]:
    items = eia_articles.list_archive_articles()[offset : offset + limit]
    results: list[IngestSourceResponse] = []
    for item in items:
        try:
            doc = eia_articles.fetch_article(
                item.url, title=item.title, published_date=item.published_date
            )
            if doc is None:
                continue
            doc_id, n = store_source_document(doc, force=force)
        except Exception as exc:  # noqa: BLE001 — skip a bad article, keep going
            print(f"skip {item.url}: {exc}")
            continue
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


@app.post("/admin/ingest/eia-explained", response_model=list[IngestSourceResponse])
def admin_ingest_eia_explained(force: bool = False) -> list[IngestSourceResponse]:
    results: list[IngestSourceResponse] = []
    for url, title in eia_explained.EXPLAINED_PAGES:
        doc = eia_explained.fetch_page(url, title)
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


@app.post("/admin/backfill")
def admin_backfill() -> dict:
    """Populate energy-domain + temporal metadata for existing chunks (document-level)."""
    from app.ingestion.pipeline import compute_document_metadata

    with session_scope() as session:
        docs = session.query(Document).all()
        updated = 0
        for doc in docs:
            full_text = doc.title + ". " + " ".join(ch.text for ch in doc.chunks)
            meta = compute_document_metadata(doc.title, full_text, doc.published_date)
            for chunk in doc.chunks:
                chunk.energy_type = meta["energy_type"]
                chunk.market_layer = meta["market_layer"]
                chunk.geography = meta["geography"]
                chunk.sector = meta["sector"]
                chunk.start_year = meta["start_year"]
                chunk.end_year = meta["end_year"]
                chunk.event_start_date = meta["event_start_date"]
                chunk.event_end_date = meta["event_end_date"]
                chunk.mentioned_years = meta["mentioned_years"]
                updated += 1
        session.commit()
    return {"backfilled_chunks": updated}


@app.get("/admin/coverage")
def admin_coverage() -> dict:
    """Yearly + energy-domain coverage for the audit script."""
    with session_scope() as session:
        year_rows = session.execute(
            select(Chunk.start_year, Chunk.energy_type, Chunk.source_name,
                   Chunk.market_layer, func.count())
            .group_by(Chunk.start_year, Chunk.energy_type, Chunk.source_name, Chunk.market_layer)
        ).all()
        years: dict[int, dict] = {}
        for y, et, src, ml, cnt in year_rows:
            if y is None:
                continue
            d = years.setdefault(y, {"chunks": 0, "eia": 0, "wiki": 0, "electricity": 0, "natural_gas": 0, "retail_price": 0})
            d["chunks"] += cnt
            if src == "U.S. Energy Information Administration":
                d["eia"] += cnt
            elif src == "Wikipedia":
                d["wiki"] += cnt
            if et == "electricity":
                d["electricity"] += cnt
            if et == "natural_gas":
                d["natural_gas"] += cnt
            if ml == "retail_price":
                d["retail_price"] += cnt
        energy_types = dict(session.execute(
            select(Chunk.energy_type, func.count()).group_by(Chunk.energy_type)
        ).all())
        market_layers = dict(session.execute(
            select(Chunk.market_layer, func.count()).group_by(Chunk.market_layer)
        ).all())
        n_docs = session.scalar(select(func.count()).select_from(Document)) or 0
        n_chunks = session.scalar(select(func.count()).select_from(Chunk)) or 0
    return {
        "documents": n_docs,
        "chunks": n_chunks,
        "years": {str(y): v for y, v in sorted(years.items())},
        "energy_types": energy_types,
        "market_layers": market_layers,
    }


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
