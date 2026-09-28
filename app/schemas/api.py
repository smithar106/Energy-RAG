"""Request/response models for the HTTP API."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    # Optional explicit time period; if omitted, the agent derives it.
    time_period_start: date | None = None
    time_period_end: date | None = None


class Source(BaseModel):
    id: int
    title: str
    source: str
    url: str | None = None
    excerpt: str


class SQLResult(BaseModel):
    query: str
    rows: list[dict]
    derived: dict = Field(default_factory=dict)


class GroundingCheck(BaseModel):
    total_numeric_claims: int = 0
    grounded_numeric_claims: int = 0
    total_explanatory_claims: int = 0
    grounded_explanatory_claims: int = 0
    ungrounded: list[str] = Field(default_factory=list)
    valid: bool = True


class AskResponse(BaseModel):
    answer: str
    citations: list[Source] = Field(default_factory=list)
    sql_results: list[SQLResult] = Field(default_factory=list)
    retrieved_chunks: int = 0
    grounding: GroundingCheck = Field(default_factory=GroundingCheck)


class IngestRequest(BaseModel):
    text: str
    title: str
    source: str = "manual"
    source_url: str | None = None
    start_date: date | None = None
    end_date: date | None = None


class IngestResponse(BaseModel):
    document_id: int
    chunks: int
    embedding_model: str


class IngestEIAResponse(BaseModel):
    series_id: str
    records: int
    start: str
    end: str
