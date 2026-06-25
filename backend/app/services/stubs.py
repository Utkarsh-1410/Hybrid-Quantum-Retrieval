from hashlib import sha256
from uuid import UUID, uuid4

from app.domain.models import RetrievalCandidate
from app.indexing.models import IndexingJobSnapshot
from app.rag.generator import GenerationResult
from app.rag.retriever import RAGEvidence, RAGRetrievalResult
from app.schemas.documents import DocumentInput
from app.schemas.search import SearchFilters


class DeterministicDevelopmentEmbedder:
    """Small deterministic adapter used until a model provider is configured."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_query(self, query: str) -> tuple[float, ...]:
        digest = sha256(query.encode("utf-8")).digest()
        values = [
            (digest[index % len(digest)] / 127.5) - 1.0
            for index in range(self.dimension)
        ]
        if not any(values):
            values[0] = 1.0
        return tuple(values)


class EmptyLexicalRetriever:
    async def search(
        self, query: str, limit: int, filters: SearchFilters
    ) -> list[RetrievalCandidate]:
        return []


class EmptyDenseRetriever:
    async def search(
        self,
        query_embedding: tuple[float, ...],
        limit: int,
        filters: SearchFilters,
    ) -> list[RetrievalCandidate]:
        return []


class PendingDocumentIndexer:
    async def submit(self, documents: list[DocumentInput]) -> str:
        return str(uuid4())

    async def get_job(
        self,
        job_id: UUID,
    ) -> IndexingJobSnapshot | None:
        return None


class EmptyRAGRetriever:
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> RAGRetrievalResult:
        return RAGRetrievalResult(query=query, evidence=[])


class GroundedPlaceholderGenerator:
    async def generate(
        self,
        query: str,
        evidence: list[RAGEvidence],
    ) -> GenerationResult:
        if not evidence:
            return GenerationResult(
                answer=(
                    "No indexed evidence is available yet. Submit documents "
                    "to /index before asking a question."
                ),
                cited_ids=(),
            )
        return GenerationResult(
            answer="Answer generation provider is not configured.",
            cited_ids=(),
        )
