import asyncio
from dataclasses import dataclass, replace
from uuid import UUID

from app.domain.models import RetrievalCandidate
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, DocumentEmbedder
from app.schemas.search import SearchFilters


@dataclass(frozen=True, slots=True)
class HybridConfig:
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    candidate_multiplier: int = 5

    def __post_init__(self) -> None:
        if abs(self.bm25_weight + self.dense_weight - 1.0) > 1e-9:
            raise ValueError("retrieval weights must sum to 1")
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    query_embedding: tuple[float, ...]
    candidates: list[tuple[RetrievalCandidate, float]]


class HybridRetriever:
    """Coordinate query embedding, BM25 retrieval, FAISS search, and fusion."""

    def __init__(
        self,
        *,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        embedder: DocumentEmbedder,
        config: HybridConfig | None = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.config = config or HybridConfig()

    @property
    def document_count(self) -> int:
        return max(self.bm25.document_count, self.dense.document_count)

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[tuple[RetrievalCandidate, float]]:
        result = await self.retrieve(query, limit=limit, filters=filters)
        return result.candidates

    async def retrieve(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> HybridRetrievalResult:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not query.strip():
            raise ValueError("query cannot be empty")
        if self.document_count == 0:
            return HybridRetrievalResult(
                query_embedding=(),
                candidates=[],
            )
        active_filters = filters or SearchFilters()
        candidate_limit = max(
            limit,
            limit * self.config.candidate_multiplier,
        )
        query_embedding, lexical = await asyncio.gather(
            self.embedder.embed_query(query),
            self.bm25.search(query, candidate_limit, active_filters),
        )
        dense = await self.dense.search(
            query_embedding,
            candidate_limit,
            active_filters,
        )
        return HybridRetrievalResult(
            query_embedding=query_embedding,
            candidates=fuse_candidates(
                lexical,
                dense,
                bm25_weight=self.config.bm25_weight,
                dense_weight=self.config.dense_weight,
            )[:limit],
        )


def _normalize(scores: dict[UUID, float]) -> dict[UUID, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        return {key: 1.0 if maximum > 0 else 0.0 for key in scores}
    scale = maximum - minimum
    return {key: (value - minimum) / scale for key, value in scores.items()}


def fuse_candidates(
    lexical: list[RetrievalCandidate],
    dense: list[RetrievalCandidate],
    *,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> list[tuple[RetrievalCandidate, float]]:
    if abs(bm25_weight + dense_weight - 1.0) > 1e-9:
        raise ValueError("retrieval weights must sum to 1")

    candidates: dict[UUID, RetrievalCandidate] = {}
    lexical_scores: dict[UUID, float] = {}
    dense_scores: dict[UUID, float] = {}

    for candidate in lexical:
        candidates[candidate.chunk_id] = candidate
        lexical_scores[candidate.chunk_id] = candidate.bm25_score

    for candidate in dense:
        existing = candidates.get(candidate.chunk_id)
        candidates[candidate.chunk_id] = (
            replace(existing, embedding=candidate.embedding)
            if existing is not None and candidate.embedding
            else candidate
        )
        dense_scores[candidate.chunk_id] = candidate.dense_score

    normalized_lexical = _normalize(lexical_scores)
    normalized_dense = _normalize(dense_scores)
    fused: list[tuple[RetrievalCandidate, float]] = []

    for chunk_id, candidate in candidates.items():
        bm25 = normalized_lexical.get(chunk_id, 0.0)
        dense_score = normalized_dense.get(chunk_id, 0.0)
        hydrated = replace(
            candidate,
            bm25_score=bm25,
            dense_score=dense_score,
        )
        score = bm25_weight * bm25 + dense_weight * dense_score
        fused.append((hydrated, score))

    return sorted(fused, key=lambda item: item[1], reverse=True)
