"""Request/response models for the HTTP API."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


# ── Ask ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    time_period_start: date | None = None
    time_period_end: date | None = None
    include_trace: bool = True


class Source(BaseModel):
    """A citation, injected programmatically from a retrieved chunk record."""

    index: int                       # matches the [index] marker in the answer
    chunk_id: int
    document_id: int
    title: str
    source_name: str
    source_type: str
    url: str | None = None
    published_date: date | None = None
    start_year: int | None = None
    end_year: int | None = None
    section: str | None = None
    excerpt: str


class GroundingCheck(BaseModel):
    total_numeric_claims: int = 0
    grounded_numeric_claims: int = 0
    total_explanatory_claims: int = 0
    grounded_explanatory_claims: int = 0
    ungrounded: list[str] = Field(default_factory=list)
    citations_used: list[int] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    citations_valid: bool = True
    sufficient_evidence: bool = True
    notes: list[str] = Field(default_factory=list)
    valid: bool = True


class ChunkTrace(BaseModel):
    rank: int
    chunk_id: int | None = None
    document_id: int | None = None
    document_title: str
    section: str | None = None
    source_name: str
    source_type: str
    source_url: str
    published_date: date | None = None
    start_year: int | None = None
    end_year: int | None = None
    semantic_similarity: float
    temporal_score: float
    temporal_reason: str | None = None
    authority_score: float
    final_score: float
    text: str
    passed_to_llm: bool = False


class RAGTrace(BaseModel):
    question: str
    requested_period: dict
    retrieval_query: str | None = None
    embedding_model: str
    embedding_dim: int
    query_embedding: list[float] = Field(default_factory=list)
    candidate_count: int = 0
    top_n: int = 0
    ranking_weights: dict = Field(default_factory=dict)
    ranking_formula: str = ""
    generated_retrieval_query: str | None = None
    chunks: list[ChunkTrace] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    citations: list[Source] = Field(default_factory=list)
    sql_results: list[dict] = Field(default_factory=list)
    retrieved_chunks: int = 0
    grounding: GroundingCheck = Field(default_factory=GroundingCheck)
    trace: RAGTrace | None = None


# ── Retrieval debug ────────────────────────────────────────────────────────
class RetrieveRequest(BaseModel):
    query: str
    time_period_start: date | None = None
    time_period_end: date | None = None
    generate_query: bool = False   # if true, rewrite via DeepSeek first
    candidate_k: int | None = None
    top_n: int | None = None


class RetrieveResponse(BaseModel):
    trace: RAGTrace


# ── Ingestion ──────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    text: str
    title: str
    source_name: str
    source_type: str
    source_url: str
    published_date: date | None = None
    start_year: int | None = None
    end_year: int | None = None
    section: str | None = None
    force: bool = False


class IngestResponse(BaseModel):
    document_id: int
    chunks: int
    embedding_model: str
    source_url: str


class IngestEIAResponse(BaseModel):
    series_id: str
    records: int
    start: str
    end: str


class IngestSourceResponse(BaseModel):
    title: str
    source_name: str
    source_url: str
    document_id: int
    chunks: int
    published_date: date | None = None
    start_year: int | None = None
    end_year: int | None = None


class ResetResponse(BaseModel):
    dropped: list[str]
    recreated: list[str]
