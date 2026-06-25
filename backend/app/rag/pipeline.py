from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.rag.generator import GenerationResult
from app.rag.retriever import RAGEvidence, RAGRetrievalResult
from app.schemas.search import SearchFilters


class EvidenceRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> RAGRetrievalResult: ...


class AnswerGenerator(Protocol):
    async def generate(
        self,
        query: str,
        evidence: list[RAGEvidence],
    ) -> GenerationResult: ...


@dataclass(frozen=True, slots=True)
class RAGPipelineConfig:
    minimum_evidence: int = 3
    evidence_weight: float = 0.65
    coverage_weight: float = 0.15
    citation_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.minimum_evidence < 1:
            raise ValueError("minimum_evidence must be positive")
        weights = (
            self.evidence_weight,
            self.coverage_weight,
            self.citation_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("confidence weights cannot be negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("confidence weights must sum to 1")


@dataclass(frozen=True, slots=True)
class RAGPipelineResult:
    answer: str
    citations: list[RAGEvidence]
    evidence: list[RAGEvidence]
    confidence: float


class RAGPipeline:
    """End-to-end retrieval, quantum ranking, prompting, and generation."""

    def __init__(
        self,
        *,
        retriever: EvidenceRetriever,
        generator: AnswerGenerator,
        config: RAGPipelineConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.config = config or RAGPipelineConfig()

    async def ask(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> RAGPipelineResult:
        retrieval = await self.retriever.retrieve(
            query,
            top_k=top_k,
            filters=filters,
        )
        generation = await self.generator.generate(
            query,
            retrieval.evidence,
        )
        citation_lookup = {item.citation_id: item for item in retrieval.evidence}
        citations = [
            citation_lookup[citation_id]
            for citation_id in generation.cited_ids
            if citation_id in citation_lookup
        ]
        confidence = self.confidence(
            retrieval.evidence,
            generation,
        )
        return RAGPipelineResult(
            answer=generation.answer,
            citations=citations,
            evidence=retrieval.evidence,
            confidence=confidence,
        )

    def confidence(
        self,
        evidence: list[RAGEvidence],
        generation: GenerationResult,
    ) -> float:
        if not evidence:
            return 0.0
        top_evidence = evidence[: self.config.minimum_evidence]
        evidence_quality = sum(item.final_score for item in top_evidence) / len(
            top_evidence
        )
        coverage = min(
            1.0,
            len(evidence) / self.config.minimum_evidence,
        )
        cited = set(generation.cited_ids)
        citation_support = min(
            1.0,
            len(cited) / max(1, min(len(evidence), 2)),
        )
        confidence = (
            self.config.evidence_weight * evidence_quality
            + self.config.coverage_weight * coverage
            + self.config.citation_weight * citation_support
        )
        return round(min(1.0, max(0.0, confidence)), 4)
