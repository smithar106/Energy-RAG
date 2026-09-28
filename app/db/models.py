"""SQLAlchemy ORM models.

Two deliberately separate domains:

1. **Structured data** — ``PriceRecord``. The *quantitative truth*. Every energy
   price number the agent reports MUST originate here, via SQL. DeepSeek is
   never the source of a number.

2. **Unstructured evidence** — ``Document`` / ``Chunk``. The *historical
   explanation*. Chunks store the pgvector embedding plus the provenance and
   temporal metadata required for citation integrity and temporal retrieval.
"""
from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PriceRecord(Base):
    """A verified, structured energy price observation (the SQL side)."""

    __tablename__ = "price_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    series_id: Mapped[str] = mapped_column(String(128), index=True)
    period: Mapped[date] = mapped_column(Date, index=True)  # temporal axis
    price: Mapped[float] = mapped_column(Float)             # the verified number
    units: Mapped[str] = mapped_column(String(32))          # e.g. "$/MWh", "$/gal"
    region: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fuel: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="EIA")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Document(Base):
    """A real source document ingested into the knowledge base.

    ``published_date`` is the *source's* publication date and is kept strictly
    separate from the event window (``start_year`` / ``end_year``) carried on
    chunks — publication year does not equal event year.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    source_name: Mapped[str] = mapped_column(String(128))   # "Wikipedia", "U.S. EIA"
    source_type: Mapped[str] = mapped_column(String(64))    # "Wikipedia", "EIA Analysis"
    source_url: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A real text chunk with its pgvector embedding and full provenance.

    Provenance columns are denormalized onto the chunk so that every retrieved
    record is self-describing for citation (chunks are a read-optimized store).
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Provenance (denormalized for self-describing retrieval records).
    source_name: Mapped[str] = mapped_column(String(128))
    source_type: Mapped[str] = mapped_column(String(64))
    document_title: Mapped[str] = mapped_column(String(512))
    source_url: Mapped[str] = mapped_column(String(1024))
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Event window (inferred from source content, NOT the publication date).
    start_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    end_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    # Fine-grained temporal metadata (event dates + all mentioned years).
    event_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    event_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    mentioned_years: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # Energy-domain metadata (inferred; enables domain gating at retrieval).
    energy_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    market_layer: Mapped[str | None] = mapped_column(String(32), nullable=True)
    geography: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(32), nullable=True)

    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")
