from uuid import uuid4

import pytest

from app.domain.models import RetrievalCandidate
from app.rag.generator import GenerationResult
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RAGEvidence, RAGRetrievalResult
from app.schemas.search import SearchFilters


def evidence(
    citation_id: str,
    score: float,
) -> RAGEvidence:
    return RAGEvidence(
        citation_id=citation_id,
        candidate=RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=uuid4(),
            title=citation_id,
            content="Evidence",
        ),
        rank=int(citation_id[1:]),
        hybrid_score=score,
        quantum_score=score,
        context_score=score,
        final_score=score,
    )


class FakeRetriever:
    def __init__(self, items: list[RAGEvidence]) -> None:
        self.items = items

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> RAGRetrievalResult:
        return RAGRetrievalResult(query=query, evidence=self.items[:top_k])


class FakeGenerator:
    async def generate(
        self,
        query: str,
        evidence: list[RAGEvidence],
    ) -> GenerationResult:
        return GenerationResult(
            answer="Grounded answer [S1].",
            cited_ids=("S1",),
        )


@pytest.mark.asyncio
async def test_pipeline_returns_answer_citations_and_confidence() -> None:
    items = [evidence("S1", 0.9), evidence("S2", 0.7)]
    pipeline = RAGPipeline(
        retriever=FakeRetriever(items),
        generator=FakeGenerator(),
    )

    result = await pipeline.ask("Question", top_k=2)

    assert result.answer == "Grounded answer [S1]."
    assert [item.citation_id for item in result.citations] == ["S1"]
    assert 0 < result.confidence <= 1


@pytest.mark.asyncio
async def test_pipeline_confidence_is_zero_without_evidence() -> None:
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        generator=FakeGenerator(),
    )

    result = await pipeline.ask("Question", top_k=2)

    assert result.confidence == 0.0
    assert result.citations == []
