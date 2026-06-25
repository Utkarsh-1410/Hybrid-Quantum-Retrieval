from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from math import log
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID

from app.domain.models import RetrievalCandidate
from app.schemas.search import SearchFilters

TOKEN_PATTERN = re.compile(r"\b[\w-]+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for indexing and querying."""
    return TOKEN_PATTERN.findall(text.casefold())


@dataclass(frozen=True, slots=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")


class BM25Retriever:
    """In-memory BM25 index with deterministic JSON persistence."""

    def __init__(
        self,
        documents: list[RetrievalCandidate] | None = None,
        *,
        config: BM25Config | None = None,
    ) -> None:
        self.config = config or BM25Config()
        self._lock = RLock()
        self._documents: list[RetrievalCandidate] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: list[int] = []
        self._average_document_length = 0.0
        if documents:
            self.build(documents)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def build(self, documents: list[RetrievalCandidate]) -> None:
        """Replace the current corpus and rebuild all BM25 statistics."""
        _validate_unique_chunks(documents)
        tokenized = [tokenize(_searchable_text(document)) for document in documents]
        frequencies = [Counter(tokens) for tokens in tokenized]
        document_frequencies: Counter[str] = Counter()
        for frequency in frequencies:
            document_frequencies.update(frequency.keys())
        lengths = [len(tokens) for tokens in tokenized]

        with self._lock:
            self._documents = list(documents)
            self._term_frequencies = frequencies
            self._document_frequencies = document_frequencies
            self._document_lengths = lengths
            self._average_document_length = (
                sum(lengths) / len(lengths) if lengths else 0.0
            )

    def add(self, documents: list[RetrievalCandidate]) -> None:
        """Append documents and rebuild statistics atomically."""
        if not documents:
            return
        with self._lock:
            self.build([*self._documents, *documents])

    async def search(
        self,
        query: str,
        limit: int,
        filters: SearchFilters,
    ) -> list[RetrievalCandidate]:
        return await asyncio.to_thread(
            self.search_sync,
            query,
            limit=limit,
            filters=filters,
        )

    def search_sync(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievalCandidate]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_terms = tokenize(query)
        if not query_terms:
            return []
        active_filters = filters or SearchFilters()

        with self._lock:
            document_count = len(self._documents)
            if document_count == 0:
                return []
            scores: list[tuple[float, int]] = []
            for index, document in enumerate(self._documents):
                if not matches_filters(document, active_filters):
                    continue
                score = self._score_document(
                    query_terms,
                    index=index,
                    document_count=document_count,
                )
                if score > 0:
                    scores.append((score, index))

            scores.sort(
                key=lambda item: (
                    -item[0],
                    str(self._documents[item[1]].chunk_id),
                )
            )
            return [
                replace(self._documents[index], bm25_score=float(score))
                for score, index in scores[:limit]
            ]

    def save(self, path: Path) -> None:
        """Persist corpus and configuration; statistics are rebuilt on load."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "version": 1,
                "config": asdict(self.config),
                "documents": [
                    _serialize_candidate(document) for document in self._documents
                ],
            }
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported BM25 index version")
        config = BM25Config(**payload["config"])
        documents = [
            _deserialize_candidate(document) for document in payload["documents"]
        ]
        return cls(documents, config=config)

    def _score_document(
        self,
        query_terms: list[str],
        *,
        index: int,
        document_count: int,
    ) -> float:
        frequencies = self._term_frequencies[index]
        document_length = self._document_lengths[index]
        average_length = self._average_document_length or 1.0
        score = 0.0

        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = log(
                1
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.config.k1 * (
                1 - self.config.b + self.config.b * document_length / average_length
            )
            score += (
                inverse_document_frequency
                * (term_frequency * (self.config.k1 + 1))
                / denominator
            )
        return score


def matches_filters(
    candidate: RetrievalCandidate,
    filters: SearchFilters,
) -> bool:
    metadata = candidate.metadata
    if filters.language is not None and metadata.get("language") != filters.language:
        return False
    if filters.source is not None and metadata.get("source") != filters.source:
        return False
    return all(metadata.get(key) == value for key, value in filters.metadata.items())


def _searchable_text(candidate: RetrievalCandidate) -> str:
    return f"{candidate.title} {candidate.content}".strip()


def _validate_unique_chunks(documents: list[RetrievalCandidate]) -> None:
    chunk_ids = [document.chunk_id for document in documents]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk IDs must be unique")


def _serialize_candidate(candidate: RetrievalCandidate) -> dict[str, Any]:
    return {
        "chunk_id": str(candidate.chunk_id),
        "document_id": str(candidate.document_id),
        "title": candidate.title,
        "content": candidate.content,
        "url": candidate.url,
        "embedding": list(candidate.embedding),
        "metadata": candidate.metadata,
    }


def _deserialize_candidate(payload: dict[str, Any]) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=UUID(payload["chunk_id"]),
        document_id=UUID(payload["document_id"]),
        title=str(payload["title"]),
        content=str(payload["content"]),
        url=payload.get("url"),
        embedding=tuple(float(value) for value in payload["embedding"]),
        metadata=dict(payload["metadata"]),
    )
