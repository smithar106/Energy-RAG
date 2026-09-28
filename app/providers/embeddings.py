"""Embedding provider interface + local sentence-transformers implementation.

The embedding model is:
  - loaded ONCE per process (singleton), and
  - reused for every ingestion and query.

This is the single place to swap the embedding model. The model name is
``EMBEDDING_MODEL`` (default ``BAAI/bge-small-en-v1.5``) and its dimensionality
is expected to match the pgvector column dimension (384).

BGE models want queries prefixed so they are treated as short passages.
See: https://huggingface.co/BAAI/bge-small-en-v1.5
"""
from __future__ import annotations

import threading
from typing import Protocol, Sequence

from app.config import get_settings

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingProvider(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddings:
    """Local embeddings via sentence-transformers (BGE family by default)."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import

        settings = get_settings()
        self.dim = settings.embedding_dim
        self._model = SentenceTransformer(settings.embedding_model)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        # BGE retrieval queries benefit from a short instruction prefix.
        embeddings = self._model.encode(
            [_QUERY_PREFIX + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings[0].tolist()


_embeddings: EmbeddingProvider | None = None
_lock = threading.Lock()


def get_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide singleton embedding provider."""
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = SentenceTransformerEmbeddings()
    return _embeddings
