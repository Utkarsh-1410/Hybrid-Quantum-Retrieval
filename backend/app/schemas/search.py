from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class SearchFilters(BaseModel):
    language: str | None = Field(default=None, min_length=2, max_length=16)
    source: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class ScoreExplanation(BaseModel):
    bm25: float
    dense: float
    hybrid: float
    quantum: float
    context: float
    final: float
    formula: str


class SearchResult(BaseModel):
    rank: int
    document_id: UUID
    chunk_id: UUID
    title: str
    snippet: str
    url: HttpUrl | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    scores: ScoreExplanation


class SearchResponse(BaseModel):
    request_id: UUID
    query: str
    total: int
    latency_ms: int
    results: list[SearchResult]