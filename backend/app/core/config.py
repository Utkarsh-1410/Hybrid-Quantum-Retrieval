from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Hybrid Quantum-Classical Search"
    app_env: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    database_url: str = (
        "postgresql+asyncpg://search:search@localhost:5432/hybrid_search"
    )
    database_pool_size: int = Field(default=10, ge=1)
    database_max_overflow: int = Field(default=20, ge=0)

    cors_origins: list[str] = ["http://localhost:3000"]
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, ge=1)
    bm25_index_path: Path = Path("../data/faiss/arxiv/bm25.json")
    faiss_index_path: Path = Path("../data/faiss/arxiv/documents.index")

    default_top_k: int = Field(default=10, ge=1)
    max_top_k: int = Field(default=100, ge=1)
    bm25_weight: float = Field(default=0.4, ge=0, le=1)
    dense_weight: float = Field(default=0.6, ge=0, le=1)
    ranker_hybrid_weight: float = Field(default=0.35, ge=0, le=1)
    quantum_weight: float = Field(default=0.45, ge=0, le=1)
    context_weight: float = Field(default=0.20, ge=0, le=1)

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    rag_max_chars_per_source: int = Field(default=4000, ge=1)
    rag_max_context_chars: int = Field(default=16000, ge=1)
    rag_minimum_evidence: int = Field(default=3, ge=1)
    indexing_batch_size: int = Field(default=64, ge=1, le=1000)

    @field_validator("api_v1_prefix")
    @classmethod
    def validate_prefix(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("API prefix must start with '/'")
        return value.rstrip("/")

    @field_validator("dense_weight")
    @classmethod
    def validate_retrieval_weights(cls, value: float, info: object) -> float:
        data = getattr(info, "data", {})
        bm25_weight = data.get("bm25_weight")
        if bm25_weight is not None and abs(bm25_weight + value - 1.0) > 1e-9:
            raise ValueError("BM25 and dense weights must sum to 1")
        return value

    @field_validator("context_weight")
    @classmethod
    def validate_ranking_weights(cls, value: float, info: object) -> float:
        data = getattr(info, "data", {})
        ranker_hybrid_weight = data.get("ranker_hybrid_weight")
        quantum_weight = data.get("quantum_weight")
        if (
            ranker_hybrid_weight is not None
            and quantum_weight is not None
            and abs(ranker_hybrid_weight + quantum_weight + value - 1.0) > 1e-9
        ):
            raise ValueError("Ranker weights must sum to 1")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
