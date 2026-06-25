from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import select

from app.db.models import (
    Document,
    DocumentChunk,
    IndexingJob,
    IndexingJobStatus,
)
from app.db.session import Database
from app.indexing.models import IndexedDocument, IndexingJobSnapshot


class IndexRepository(Protocol):
    async def create_job(
        self,
        job_id: UUID,
        *,
        requested_count: int,
    ) -> None: ...

    async def mark_running(self, job_id: UUID) -> None: ...

    async def existing_hashes(self, hashes: set[str]) -> set[str]: ...

    async def persist_batch(
        self,
        documents: list[IndexedDocument],
    ) -> None: ...

    async def mark_completed(
        self,
        job_id: UUID,
        *,
        indexed_count: int,
        skipped_count: int,
        index_version: str,
    ) -> None: ...

    async def mark_failed(
        self,
        job_id: UUID,
        *,
        indexed_count: int,
        skipped_count: int,
        failed_count: int,
        error_message: str,
    ) -> None: ...

    async def get_job(self, job_id: UUID) -> IndexingJobSnapshot | None: ...


class PostgresIndexRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def create_job(
        self,
        job_id: UUID,
        *,
        requested_count: int,
    ) -> None:
        async with self.database.session_factory() as session:
            session.add(
                IndexingJob(
                    id=job_id,
                    status=IndexingJobStatus.PENDING,
                    requested_count=requested_count,
                    indexed_count=0,
                    failed_count=0,
                    skipped_count=0,
                )
            )
            await session.commit()

    async def mark_running(self, job_id: UUID) -> None:
        async with self.database.session_factory() as session:
            job = await session.get(IndexingJob, job_id)
            if job is None:
                raise KeyError(f"indexing job {job_id} does not exist")
            job.status = IndexingJobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            await session.commit()

    async def existing_hashes(self, hashes: set[str]) -> set[str]:
        if not hashes:
            return set()
        async with self.database.session_factory() as session:
            rows = await session.scalars(
                select(Document.content_hash).where(Document.content_hash.in_(hashes))
            )
            return set(rows)

    async def persist_batch(
        self,
        documents: list[IndexedDocument],
    ) -> None:
        async with self.database.session_factory() as session:
            for item in documents:
                prepared = item.prepared
                source = prepared.input
                document = Document(
                    id=prepared.candidate.document_id,
                    external_id=source.external_id,
                    canonical_url=(
                        str(source.canonical_url)
                        if source.canonical_url is not None
                        else None
                    ),
                    title=source.title,
                    content=source.content,
                    language=source.language,
                    source=source.source,
                    document_metadata=source.metadata,
                    content_hash=prepared.content_hash,
                    is_active=True,
                )
                chunk = DocumentChunk(
                    id=prepared.candidate.chunk_id,
                    document_id=prepared.candidate.document_id,
                    chunk_index=0,
                    content=source.content,
                    token_count=len(source.content.split()),
                    chunk_metadata=source.metadata,
                    embedding=list(item.embedding),
                    embedding_model=item.embedding_model,
                    faiss_id=item.faiss_id,
                )
                session.add(document)
                session.add(chunk)
            await session.commit()

    async def mark_completed(
        self,
        job_id: UUID,
        *,
        indexed_count: int,
        skipped_count: int,
        index_version: str,
    ) -> None:
        async with self.database.session_factory() as session:
            job = await session.get(IndexingJob, job_id)
            if job is None:
                raise KeyError(f"indexing job {job_id} does not exist")
            job.status = IndexingJobStatus.COMPLETED
            job.indexed_count = indexed_count
            job.skipped_count = skipped_count
            job.index_version = index_version
            job.completed_at = datetime.now(UTC)
            await session.commit()

    async def mark_failed(
        self,
        job_id: UUID,
        *,
        indexed_count: int,
        skipped_count: int,
        failed_count: int,
        error_message: str,
    ) -> None:
        async with self.database.session_factory() as session:
            job = await session.get(IndexingJob, job_id)
            if job is None:
                return
            job.status = IndexingJobStatus.FAILED
            job.indexed_count = indexed_count
            job.skipped_count = skipped_count
            job.failed_count = failed_count
            job.error_message = error_message[:4000]
            job.completed_at = datetime.now(UTC)
            await session.commit()

    async def get_job(self, job_id: UUID) -> IndexingJobSnapshot | None:
        async with self.database.session_factory() as session:
            job = await session.get(IndexingJob, job_id)
            return _snapshot(job) if job is not None else None


class InMemoryIndexRepository:
    """Deterministic repository used by tests and local isolated workflows."""

    def __init__(self) -> None:
        self.jobs: dict[UUID, IndexingJobSnapshot] = {}
        self.hashes: set[str] = set()
        self.documents: list[IndexedDocument] = []

    async def create_job(
        self,
        job_id: UUID,
        *,
        requested_count: int,
    ) -> None:
        self.jobs[job_id] = IndexingJobSnapshot(
            job_id=job_id,
            status=IndexingJobStatus.PENDING.value,
            requested_count=requested_count,
            indexed_count=0,
            skipped_count=0,
            failed_count=0,
            created_at=datetime.now(UTC),
        )

    async def mark_running(self, job_id: UUID) -> None:
        current = self.jobs[job_id]
        self.jobs[job_id] = replace(
            current,
            status=IndexingJobStatus.RUNNING.value,
            started_at=datetime.now(UTC),
        )

    async def existing_hashes(self, hashes: set[str]) -> set[str]:
        return self.hashes & hashes

    async def persist_batch(
        self,
        documents: list[IndexedDocument],
    ) -> None:
        self.documents.extend(documents)
        self.hashes.update(item.prepared.content_hash for item in documents)

    async def mark_completed(
        self,
        job_id: UUID,
        *,
        indexed_count: int,
        skipped_count: int,
        index_version: str,
    ) -> None:
        current = self.jobs[job_id]
        self.jobs[job_id] = IndexingJobSnapshot(
            job_id=job_id,
            status=IndexingJobStatus.COMPLETED.value,
            requested_count=current.requested_count,
            indexed_count=indexed_count,
            skipped_count=skipped_count,
            failed_count=0,
            index_version=index_version,
            created_at=current.created_at,
            started_at=current.started_at,
            completed_at=datetime.now(UTC),
        )

    async def mark_failed(
        self,
        job_id: UUID,
        *,
        indexed_count: int,
        skipped_count: int,
        failed_count: int,
        error_message: str,
    ) -> None:
        current = self.jobs[job_id]
        self.jobs[job_id] = IndexingJobSnapshot(
            job_id=job_id,
            status=IndexingJobStatus.FAILED.value,
            requested_count=current.requested_count,
            indexed_count=indexed_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            error_message=error_message,
            created_at=current.created_at,
            started_at=current.started_at,
            completed_at=datetime.now(UTC),
        )

    async def get_job(self, job_id: UUID) -> IndexingJobSnapshot | None:
        return self.jobs.get(job_id)


def _snapshot(job: IndexingJob) -> IndexingJobSnapshot:
    return IndexingJobSnapshot(
        job_id=job.id,
        status=job.status.value,
        requested_count=job.requested_count,
        indexed_count=job.indexed_count,
        skipped_count=job.skipped_count,
        failed_count=job.failed_count,
        index_version=job.index_version,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
