from typing import Protocol
from uuid import UUID

from app.domain.models import RetrievalCandidate
from app.indexing.models import IndexingJobSnapshot
from app.schemas.documents import DocumentInput
from app.schemas.search import SearchFilters


class Embedder(Protocol):
    async def embed_query(self, query: str) -> tuple[float, ...]: ...


class LexicalRetriever(Protocol):
    async def search(
        self, query: str, limit: int, filters: SearchFilters
    ) -> list[RetrievalCandidate]: ...


class DenseRetriever(Protocol):
    async def search(
        self,
        query_embedding: tuple[float, ...],
        limit: int,
        filters: SearchFilters,
    ) -> list[RetrievalCandidate]: ...


class DocumentIndexer(Protocol):
    async def submit(self, documents: list[DocumentInput]) -> str: ...

    async def get_job(
        self,
        job_id: UUID,
    ) -> IndexingJobSnapshot | None: ...
