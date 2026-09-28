"""FastAPI application — the HTTP entrypoint.

The embedding model is loaded ONCE at startup (lifespan) and reused for every
ingestion/query. The full pipeline is::

    question
      -> DeepSeek agent (tool orchestration)
      -> { SQL numbers } + { pgvector evidence }
      -> DeepSeek grounded synthesis
      -> grounding validator
      -> cited response
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.agent.orchestrator import AgentOrchestrator
from app.config import get_settings
from app.db.base import init_db
from app.ingestion.eia import EIAClient
from app.ingestion.pipeline import ingest_document, ingest_price_points
from app.providers.embeddings import get_embedding_provider
from app.retrieval.temporal import parse_time_period
from app.schemas.api import (
    AskRequest,
    AskResponse,
    IngestEIAResponse,
    IngestRequest,
    IngestResponse,
)
from app.synthesis.citations import build_citations
from app.synthesis.grounding import validate_grounding
from app.synthesis.synthesizer import Synthesizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.database_url:
        init_db()
    # Warm the embedding model once so the first request is not slow.
    get_embedding_provider()
    yield


app = FastAPI(title="Energy-RAG", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "llm_provider": "deepseek",
        "llm_model": settings.deepseek_model,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # 1. Agent orchestration (gathers SQL + evidence).
    agent = AgentOrchestrator()
    result = agent.run(question)

    # 2. Grounded synthesis.
    synthesizer = Synthesizer()
    answer = synthesizer.synthesize(question, result.sql_results, result.evidence)

    # 3. Grounding validation.
    grounding = validate_grounding(answer, result.sql_results, result.evidence)

    # 4. Cited response.
    citations = build_citations(result.evidence)

    return AskResponse(
        answer=answer,
        citations=citations,
        sql_results=result.sql_results,
        retrieved_chunks=len(result.evidence),
        grounding=grounding,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    doc_id, n_chunks = ingest_document(
        text=req.text,
        title=req.title,
        source=req.source,
        source_url=req.source_url,
        start_date=req.start_date,
        end_date=req.end_date,
    )
    settings = get_settings()
    return IngestResponse(
        document_id=doc_id,
        chunks=n_chunks,
        embedding_model=settings.embedding_model,
    )


@app.post("/ingest/eia", response_model=IngestEIAResponse)
def ingest_eia(
    series_id: str,
    length: int = 5000,
    region: str | None = None,
    fuel: str | None = None,
) -> IngestEIAResponse:
    client = EIAClient()
    points = client.fetch_series(series_id, length=length)
    if not points:
        raise HTTPException(status_code=404, detail="no data for series")
    ingest_price_points(
        series_id=series_id,
        points=points,
        region=region,
        fuel=fuel,
    )
    return IngestEIAResponse(
        series_id=series_id,
        records=len(points),
        start=points[0].period,
        end=points[-1].period,
    )
