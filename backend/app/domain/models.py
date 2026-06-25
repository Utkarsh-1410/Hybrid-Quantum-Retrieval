from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    chunk_id: UUID
    document_id: UUID
    title: str
    content: str
    url: str | None = None
    embedding: tuple[float, ...] = ()
    bm25_score: float = 0.0
    dense_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: RetrievalCandidate
    rank: int
    hybrid_score: float
    quantum_score: float
    context_score: float
    final_score: float
