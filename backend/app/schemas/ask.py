from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

from app.schemas.search import SearchFilters, SearchResult


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class Citation(BaseModel):
    citation_id: str
    rank: int = Field(ge=1)
    document_id: UUID
    chunk_id: UUID
    title: str
    url: HttpUrl | None = None
    excerpt: str
    score: float = Field(ge=0, le=1)


class AskResponse(BaseModel):
    request_id: UUID
    answer: str
    citations: list[Citation]
    sources: list[SearchResult]
    confidence: float = Field(ge=0, le=1)
    latency_ms: int
