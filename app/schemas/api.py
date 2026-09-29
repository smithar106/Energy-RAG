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
    event_start_date: date | None = None
    event_end_date: date | None = None
    energy_type: str | None = None
    market_layer: str | None = None
    geography: str | None = None
    sector: str | None = None
    semantic_similarity: float = 0.0
    lexical_score: float = 0.0
    temporal_score: float = 0.0
    temporal_reason: str | None = None
    temporal_coverage: float | None = None
    temporal_specificity: float | None = None
    chunk_span: int | None = None
    domain_score: float = 0.0
    metric_score: float = 0.0
    geography_score: float = 0.0
    authority_score: float = 0.0
    final_score: float = 0.0
    gate: str | None = None
    failure_reason: str | None = None
    from_vector: bool = False
    from_lexical: bool = False
    passed_to_llm: bool = False
    text: str


class RAGTrace(BaseModel):
    question: str
    intent: dict = Field(default_factory=dict)
    requested_period: dict = Field(default_factory=dict)
    queries: list[str] = Field(default_factory=list)
    embedding_model: str
    embedding_dim: int
    query_embedding: list[float] = Field(default_factory=list)
    vector_count: int = 0
    lexical_count: int = 0
    candidate_count: int = 0      # retrieved (merged) candidates
    gate_passed: int = 0          # passed the deterministic evidence gate
    causally_useful: int = 0      # passed the causal-usefulness filter
    evidence_supplied: int = 0    # final evidence sent to DeepSeek
    top_n: int = 0
    ranking_weights: dict = Field(default_factory=dict)
    ranking_formula: str = ""
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
