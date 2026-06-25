from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.domain.models import RetrievalCandidate
from app.schemas.documents import DocumentInput


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    input: DocumentInput
    candidate: RetrievalCandidate
    content_hash: str


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    prepared: PreparedDocument
    embedding: tuple[float, ...]
    faiss_id: int
    embedding_model: str


@dataclass(frozen=True, slots=True)
class IndexingJobSnapshot:
    job_id: UUID
    status: str
    requested_count: int
    indexed_count: int
    skipped_count: int
    failed_count: int
    index_version: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
