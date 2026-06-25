from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain.models import RetrievalCandidate
from app.quantum.contextual_ranker import QuantumContextualRanker
from app.retrieval.hybrid import HybridRetrievalResult
from app.schemas.search import SearchFilters


class HybridRetrievalBackend(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> HybridRetrievalResult: ...


@dataclass(frozen=True, slots=True)
class RAGEvidence:
    citation_id: str
    candidate: RetrievalCandidate
    rank: int
    hybrid_score: float
    quantum_score: float
    context_score: float
    final_score: float

    def prompt_block(self, *, max_chars: int) -> str:
        content = self.candidate.content.strip()
        if len(content) > max_chars:
            content = f"{content[: max_chars - 3].rstrip()}..."
        location = f"\nURL: {self.candidate.url}" if self.candidate.url else ""
        return (
            f"[{self.citation_id}]\n"
            f"Title: {self.candidate.title}{location}\n"
            f"Retrieval score: {self.final_score:.4f}\n"
            f"Content: {content}"
        )


@dataclass(frozen=True, slots=True)
class RAGRetrievalResult:
    query: str
    evidence: list[RAGEvidence]


@dataclass(frozen=True, slots=True)
class RAGRetrieverConfig:
    candidate_pool_multiplier: int = 5
    max_per_document: int = 2

    def __post_init__(self) -> None:
        if self.candidate_pool_multiplier < 1:
            raise ValueError("candidate_pool_multiplier must be positive")
        if self.max_per_document < 1:
            raise ValueError("max_per_document must be positive")


class RAGRetriever:
    """Run hybrid retrieval, quantum re-ranking, and evidence diversification."""

    def __init__(
        self,
        *,
        hybrid_retriever: HybridRetrievalBackend,
        quantum_ranker: QuantumContextualRanker,
        config: RAGRetrieverConfig | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.quantum_ranker = quantum_ranker
        self.config = config or RAGRetrieverConfig()

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> RAGRetrievalResult:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        candidate_pool = top_k * self.config.candidate_pool_multiplier
        hybrid = await self.hybrid_retriever.retrieve(
            query,
            limit=candidate_pool,
            filters=filters,
        )
        ranked = self.quantum_ranker.rank(
            hybrid.query_embedding,
            hybrid.candidates,
            top_k=len(hybrid.candidates) or top_k,
        )
        selected = []
        document_counts: dict[object, int] = {}
        for result in ranked:
            document_id = result.candidate.document_id
            count = document_counts.get(document_id, 0)
            if count >= self.config.max_per_document:
                continue
            document_counts[document_id] = count + 1
            selected.append(result)
            if len(selected) == top_k:
                break

        evidence = [
            RAGEvidence(
                citation_id=f"S{rank}",
                candidate=result.candidate,
                rank=rank,
                hybrid_score=result.hybrid_score,
                quantum_score=result.quantum_score,
                context_score=result.context_score,
                final_score=result.final_score,
            )
            for rank, result in enumerate(selected, start=1)
        ]
        return RAGRetrievalResult(query=query, evidence=evidence)
