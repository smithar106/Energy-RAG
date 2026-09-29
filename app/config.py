"""Application configuration loaded from environment variables.

All secrets and tunables live here so the rest of the codebase never reads
``os.environ`` directly.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM — DeepSeek is the ONLY hosted LLM provider.
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Embeddings — local, no paid embeddings API.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # bge-small-en-v1.5 -> 384 dims

    # Database.
    database_url: str = ""

    # Structured energy data.
    eia_api_key: str = ""

    # Model cache location (baked into the image on Railway).
    hf_home: str = "/app/.hf_cache"

    # Security / resilience.
    admin_api_key: str = ""        # guards /admin/* (fail-closed when empty)
    rate_limit_max: int = 20       # max /ask requests per window per IP (0 = off)
    rate_limit_window: int = 60    # window length in seconds

    # Retrieval knobs.
    retrieval_top_k: int = 40   # pgvector candidate pool before ranking (stage 1)
    rerank_top_n: int = 5       # accepted evidence passed to synthesis (stage 2)


@lru_cache
def get_settings() -> Settings:
    return Settings()
