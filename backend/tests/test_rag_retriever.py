from uuid import uuid4

import pytest

from app.domain.models import RetrievalCandidate
from app.quantum.contextual_ranker import QuantumContextualRanker
from app.rag.retriever import RAGRetriever, RAGRetrieverConfig
from app.retrieval.hybrid import HybridRetrievalResult
from app.schemas.search import SearchFilters


class FakeHybridRetriever:
    def __init__(
        self,
        candidates: list[tuple[RetrievalCandidate, float]],
    ) -> None:
        self.candidates = candidates
        self.limit = 0

    async def retrieve(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> HybridRetrievalResult:
        self.limit = limit
        return HybridRetrievalResult(
            query_embedding=(1.0, 0.0),
            candidates=self.candidates,
        )


def document(
    title: str,
    embedding: tuple[float, ...],
    *,
    document_id: object | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=document_id or uuid4(),  # type: ignore[arg-type]
        title=title,
        content=f"{title} content",
        embedding=embedding,
    )


@pytest.mark.asyncio
async def test_rag_retriever_hybrid_then_quantum_ranks_evidence() -> None:
    aligned = document("Aligned", (1.0, 0.0))
    orthogonal = document("Orthogonal", (0.0, 1.0))
    backend = FakeHybridRetriever([(orthogonal, 0.8), (aligned, 0.6)])
    retriever = RAGRetriever(
        hybrid_retriever=backend,
        quantum_ranker=QuantumContextualRanker(),
        config=RAGRetrieverConfig(candidate_pool_multiplier=4),
    )

    result = await retriever.retrieve("query", top_k=2)

    assert backend.limit == 8
    assert result.evidence[0].candidate.chunk_id == aligned.chunk_id
    assert [item.citation_id for item in result.evidence] == ["S1", "S2"]


@pytest.mark.asyncio
async def test_rag_retriever_limits_chunks_per_document() -> None:
    shared_id = uuid4()
    first = document("First", (1.0, 0.0), document_id=shared_id)
    second = document("Second", (0.9, 0.1), document_id=shared_id)
    other = document("Other", (0.8, 0.2))
    backend = FakeHybridRetriever([(first, 0.9), (second, 0.8), (other, 0.7)])
    retriever = RAGRetriever(
        hybrid_retriever=backend,
        quantum_ranker=QuantumContextualRanker(),
        config=RAGRetrieverConfig(max_per_document=1),
    )

    result = await retriever.retrieve("query", top_k=2)

    assert len(result.evidence) == 2
    assert len({item.candidate.document_id for item in result.evidence}) == 2
