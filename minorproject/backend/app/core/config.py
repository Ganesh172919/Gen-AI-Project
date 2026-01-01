"""FaithForge configuration management.

Uses pydantic-settings for env-based configuration. All settings can be
overridden via environment variables or a .env file.
"""

from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class VectorStoreType(str, Enum):
    """Supported vector store backends."""
    CHROMADB = "chromadb"
    PGVECTOR = "pgvector"


class LLMProvider(str, Enum):
    """Supported LLM API providers."""
    GROQ = "groq"
    CEREBRAS = "cerebras"
    OPENROUTER = "openrouter"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "FaithForge"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "text"  # "text" or "json"
    log_file_path: Optional[str] = None  # Set to enable file logging
    log_max_bytes: int = 10_000_000  # 10MB before rotation
    log_backup_count: int = 7  # Number of rotated backups to keep

    # ── API Keys ─────────────────────────────────────────────────────────────
    groq_api_key: Optional[str] = None
    cerebras_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.GROQ
    generator_model: str = "llama-3.3-70b-versatile"
    generator_temperature: float = 0.3
    generator_max_tokens: int = 2048

    # ── Vector Store ─────────────────────────────────────────────────────────
    vector_store_type: VectorStoreType = VectorStoreType.CHROMADB
    chromadb_path: str = "./data/chromadb"
    pgvector_url: str = "postgresql://faithforge:faithforge@localhost:5432/faithforge"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Retriever ────────────────────────────────────────────────────────────
    retriever_top_k: int = 10
    reranker_top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bm25_index_path: str = "./data/bm25_index.pkl"

    # ── Verifier ─────────────────────────────────────────────────────────────
    verifier_model_path: str = "./models/verifier"
    verifier_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    verifier_max_iterations: int = 3
    verifier_confidence_threshold: float = 0.7

    # ── Redis / Queue ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "faithforge:jobs"

    # ── OpenTelemetry ────────────────────────────────────────────────────────
    otel_exporter_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "faithforge-backend"

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
