"""SQLAlchemy ORM models.

Two deliberately separate domains:

1. **Structured data** — ``PriceRecord``. The *quantitative truth*. Every energy
   price number the agent reports MUST originate here, via SQL. DeepSeek is
   never the source of a number.

2. **Unstructured evidence** — ``Document`` / ``Chunk``. The *historical
   explanation*. Chunks carry pgvector embeddings and temporal bounds so
   retrieval can do similarity search + time-period filtering.
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
    """A source document ingested into the knowledge base."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    source: Mapped[str] = mapped_column(String(128))        # "EIA", "wikipedia", ...
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A text chunk with its pgvector embedding + temporal bounds."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # 384 dims = BAAI/bge-small-en-v1.5
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")
