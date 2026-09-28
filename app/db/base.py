"""Engine + session + schema bootstrap for PostgreSQL/pgvector.

Provides:
  - ``engine``: SQLAlchemy engine bound to ``DATABASE_URL``
  - ``SessionLocal``: session factory
  - ``init_db()``: creates the pgvector extension + tables
  - ``reset_rag_tables()``: drops and recreates ONLY the RAG tables
    (``documents`` / ``chunks``), leaving ``price_records`` untouched
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_engine = None
_SessionLocal = None


def _normalize_url(url: str) -> str:
    """Railway injects ``postgresql://`` (psycopg2 default). We use psycopg3,
    so force the driver explicitly."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is not set")
        _engine = create_engine(_normalize_url(settings.database_url), pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for a session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create the pgvector extension and all tables."""
    from app.db import models  # noqa: F401  (register mappers)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    models.Base.metadata.create_all(bind=engine)


def reset_rag_tables() -> None:
    """Drop + recreate ONLY ``documents``/``chunks``.

    Used to permanently remove synthetic/placeholder RAG content and to apply
    schema changes. ``price_records`` (real EIA data) is never touched.
    """
    from app.db import models

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    # Child table first to satisfy the FK.
    models.Chunk.__table__.drop(bind=engine, checkfirst=True)
    models.Document.__table__.drop(bind=engine, checkfirst=True)
    models.Document.__table__.create(bind=engine)
    models.Chunk.__table__.create(bind=engine)
