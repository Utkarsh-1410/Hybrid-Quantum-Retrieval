from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, cast
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from app.domain.models import RetrievalCandidate
from app.retrieval.bm25 import matches_filters
from app.schemas.search import SearchFilters

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


class DocumentEmbedder(Protocol):
    async def embed_query(self, query: str) -> tuple[float, ...]: ...

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> FloatMatrix: ...


class VectorIndex(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def size(self) -> int: ...

    def add(self, vectors: FloatMatrix) -> None: ...

    def search(
        self,
        query: FloatMatrix,
        limit: int,
    ) -> tuple[FloatMatrix, NDArray[np.int64]]: ...

    def reconstruct(self, position: int) -> FloatVector: ...

    def save(self, path: Path) -> None: ...


class SentenceTransformerEmbedder:
    """Lazy Sentence Transformer adapter suitable for API startup."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 64,
        model: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = model
        self._model_lock = RLock()

    async def embed_query(self, query: str) -> tuple[float, ...]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        matrix = await self.embed_documents([query])
        return tuple(float(value) for value in matrix[0])

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> FloatMatrix:
        if not texts:
            raise ValueError("texts cannot be empty")
        return await asyncio.to_thread(self._encode_sync, list(texts))

    def _encode_sync(self, texts: list[str]) -> FloatMatrix:
        model = self._get_model()
        encoded = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return normalize_rows(np.asarray(encoded, dtype=np.float32))

    def _get_model(self) -> Any:
        with self._model_lock:
            if self._model is None:
                module = importlib.import_module("sentence_transformers")
                self._model = module.SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
            return self._model


class FaissFlatIPIndex:
    """Exact cosine-search FAISS index over normalized vectors."""

    def __init__(self, index: Any) -> None:
        self._index = index

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @classmethod
    def create(cls, dimension: int) -> FaissFlatIPIndex:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        faiss = importlib.import_module("faiss")
        return cls(faiss.IndexFlatIP(dimension))

    @classmethod
    def load(cls, path: Path) -> FaissFlatIPIndex:
        faiss = importlib.import_module("faiss")
        return cls(faiss.read_index(str(path)))

    def add(self, vectors: FloatMatrix) -> None:
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(
        self,
        query: FloatMatrix,
        limit: int,
    ) -> tuple[FloatMatrix, NDArray[np.int64]]:
        scores, positions = self._index.search(
            np.ascontiguousarray(query, dtype=np.float32),
            limit,
        )
        return (
            np.asarray(scores, dtype=np.float32),
            np.asarray(positions, dtype=np.int64),
        )

    def reconstruct(self, position: int) -> FloatVector:
        return np.asarray(self._index.reconstruct(position), dtype=np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss = importlib.import_module("faiss")
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        faiss.write_index(self._index, str(temporary))
        temporary.replace(path)


class DenseRetriever:
    """Sentence Transformer and FAISS retrieval with metadata persistence."""

    def __init__(
        self,
        embedder: DocumentEmbedder,
        *,
        index: VectorIndex | None = None,
        documents: list[RetrievalCandidate] | None = None,
    ) -> None:
        self.embedder = embedder
        self._index = index
        self._documents = list(documents or [])
        self._lock = RLock()
        if index is not None and index.size != len(self._documents):
            raise ValueError("FAISS index and metadata sizes do not match")

    @property
    def document_count(self) -> int:
        return len(self._documents)

    async def build(self, documents: list[RetrievalCandidate]) -> None:
        if not documents:
            raise ValueError("documents cannot be empty")
        _validate_unique_chunks(documents)
        embeddings = await self.embedder.embed_documents(
            [_searchable_text(document) for document in documents]
        )
        self.build_from_embeddings(documents, embeddings)

    def build_from_embeddings(
        self,
        documents: list[RetrievalCandidate],
        embeddings: FloatMatrix,
        *,
        index: VectorIndex | None = None,
    ) -> None:
        _validate_unique_chunks(documents)
        normalized = normalize_rows(embeddings)
        if normalized.shape[0] != len(documents):
            raise ValueError("one embedding is required per document")
        vector_index = index or FaissFlatIPIndex.create(normalized.shape[1])
        if vector_index.dimension != normalized.shape[1]:
            raise ValueError("index and embedding dimensions do not match")
        if vector_index.size != 0:
            raise ValueError("build requires an empty vector index")
        vector_index.add(normalized)
        with self._lock:
            self._index = vector_index
            self._documents = list(documents)

    async def add(self, documents: list[RetrievalCandidate]) -> None:
        if not documents:
            return
        embeddings = await self.embedder.embed_documents(
            [_searchable_text(document) for document in documents]
        )
        normalized = normalize_rows(embeddings)
        with self._lock:
            if self._index is None:
                self.build_from_embeddings(documents, normalized)
                return
            existing = {document.chunk_id for document in self._documents}
            if any(document.chunk_id in existing for document in documents):
                raise ValueError("chunk IDs must be unique")
            if normalized.shape[1] != self._index.dimension:
                raise ValueError("embedding dimension changed")
            self._index.add(normalized)
            self._documents.extend(documents)

    def add_from_embeddings(
        self,
        documents: list[RetrievalCandidate],
        embeddings: FloatMatrix,
    ) -> None:
        if not documents:
            return
        normalized = normalize_rows(embeddings)
        if normalized.shape[0] != len(documents):
            raise ValueError("one embedding is required per document")
        with self._lock:
            if self._index is None:
                self.build_from_embeddings(documents, normalized)
                return
            existing = {document.chunk_id for document in self._documents}
            if any(document.chunk_id in existing for document in documents):
                raise ValueError("chunk IDs must be unique")
            if normalized.shape[1] != self._index.dimension:
                raise ValueError("embedding dimension changed")
            self._index.add(normalized)
            self._documents.extend(documents)

    async def search(
        self,
        query_embedding: tuple[float, ...],
        limit: int,
        filters: SearchFilters,
    ) -> list[RetrievalCandidate]:
        return await asyncio.to_thread(
            self.search_sync,
            query_embedding,
            limit=limit,
            filters=filters,
        )

    def search_sync(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievalCandidate]:
        if limit < 1:
            raise ValueError("limit must be positive")
        active_filters = filters or SearchFilters()
        with self._lock:
            if self._index is None or self._index.size == 0:
                return []
            query = normalize_rows(np.asarray([query_embedding], dtype=np.float32))
            if query.shape[1] != self._index.dimension:
                raise ValueError("query and index dimensions do not match")
            has_filters = bool(
                active_filters.language
                or active_filters.source
                or active_filters.metadata
            )
            search_limit = (
                self._index.size if has_filters else min(limit, self._index.size)
            )
            scores, positions = self._index.search(query, search_limit)
            results: list[RetrievalCandidate] = []
            for position, score in zip(positions[0], scores[0], strict=True):
                if position < 0:
                    continue
                document = self._documents[int(position)]
                if not matches_filters(document, active_filters):
                    continue
                embedding = tuple(
                    float(value) for value in self._index.reconstruct(int(position))
                )
                results.append(
                    RetrievalCandidate(
                        chunk_id=document.chunk_id,
                        document_id=document.document_id,
                        title=document.title,
                        content=document.content,
                        url=document.url,
                        embedding=embedding,
                        bm25_score=document.bm25_score,
                        dense_score=float(score),
                        metadata=document.metadata,
                    )
                )
                if len(results) == limit:
                    break
            return results

    def save(self, index_path: Path) -> None:
        with self._lock:
            if self._index is None:
                raise ValueError("cannot save an empty dense index")
            self._index.save(index_path)
            payload = {
                "version": 1,
                "dimension": self._index.dimension,
                "documents": [
                    _serialize_candidate(document) for document in self._documents
                ],
            }
        metadata_path = _metadata_path(index_path)
        temporary = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(metadata_path)

    @classmethod
    def load(
        cls,
        index_path: Path,
        *,
        embedder: DocumentEmbedder,
    ) -> DenseRetriever:
        payload = json.loads(_metadata_path(index_path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported dense index version")
        index = FaissFlatIPIndex.load(index_path)
        if index.dimension != int(payload["dimension"]):
            raise ValueError("FAISS index dimension does not match metadata")
        documents = [
            _deserialize_candidate(document) for document in payload["documents"]
        ]
        return cls(embedder, index=index, documents=documents)


def normalize_rows(values: FloatMatrix) -> FloatMatrix:
    if values.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embedding matrix cannot be empty")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding matrix contains a zero vector")
    return np.asarray(
        np.ascontiguousarray(values / norms),
        dtype=np.float32,
    )


def _metadata_path(index_path: Path) -> Path:
    return index_path.with_suffix(f"{index_path.suffix}.metadata.json")


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
        "metadata": candidate.metadata,
    }


def _deserialize_candidate(payload: dict[str, Any]) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=UUID(payload["chunk_id"]),
        document_id=UUID(payload["document_id"]),
        title=str(payload["title"]),
        content=str(payload["content"]),
        url=cast(str | None, payload.get("url")),
        metadata=dict(payload["metadata"]),
    )
