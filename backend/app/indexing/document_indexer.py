from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from app.domain.models import RetrievalCandidate
from app.indexing.models import (
    IndexedDocument,
    IndexingJobSnapshot,
    PreparedDocument,
)
from app.indexing.repository import IndexRepository
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, DocumentEmbedder
from app.schemas.documents import DocumentInput

logger = logging.getLogger(__name__)


class ProductionDocumentIndexer:
    """Batch and incremental indexing coordinator for durable search indexes."""

    def __init__(
        self,
        *,
        repository: IndexRepository,
        embedder: DocumentEmbedder,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        bm25_index_path: Path,
        faiss_index_path: Path,
        embedding_model: str,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.repository = repository
        self.embedder = embedder
        self.bm25 = bm25
        self.dense = dense
        self.bm25_index_path = bm25_index_path
        self.faiss_index_path = faiss_index_path
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self._index_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(self, documents: list[DocumentInput]) -> str:
        if not documents:
            raise ValueError("documents cannot be empty")
        job_id = uuid4()
        await self.repository.create_job(
            job_id,
            requested_count=len(documents),
        )
        task = asyncio.create_task(
            self._run_job(job_id, list(documents)),
            name=f"index-job-{job_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return str(job_id)

    async def get_job(self, job_id: UUID) -> IndexingJobSnapshot | None:
        return await self.repository.get_job(job_id)

    async def wait_for_job(self, job_id: UUID) -> IndexingJobSnapshot:
        matching = [
            task for task in self._tasks if task.get_name() == f"index-job-{job_id}"
        ]
        if matching:
            await asyncio.gather(*matching)
        snapshot = await self.repository.get_job(job_id)
        if snapshot is None:
            raise KeyError(f"indexing job {job_id} does not exist")
        return snapshot

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run_job(
        self,
        job_id: UUID,
        documents: list[DocumentInput],
    ) -> None:
        indexed_count = 0
        skipped_count = 0
        try:
            await self.repository.mark_running(job_id)
            async with self._index_lock:
                prepared = [self._prepare(document) for document in documents]
                unique: dict[str, PreparedDocument] = {}
                for item in prepared:
                    if item.content_hash in unique:
                        skipped_count += 1
                    else:
                        unique[item.content_hash] = item
                existing = await self.repository.existing_hashes(set(unique))
                skipped_count += len(existing)
                pending = [
                    item
                    for content_hash, item in unique.items()
                    if content_hash not in existing
                ]

                for batch in _batches(pending, self.batch_size):
                    indexed_count += await self._index_batch(batch)

                index_version = self._persist_indexes()
            await self.repository.mark_completed(
                job_id,
                indexed_count=indexed_count,
                skipped_count=skipped_count,
                index_version=index_version,
            )
        except Exception as error:
            logger.exception("Indexing job %s failed", job_id)
            failed_count = max(
                0,
                len(documents) - indexed_count - skipped_count,
            )
            await self.repository.mark_failed(
                job_id,
                indexed_count=indexed_count,
                skipped_count=skipped_count,
                failed_count=failed_count,
                error_message=str(error),
            )

    async def _index_batch(
        self,
        batch: list[PreparedDocument],
    ) -> int:
        if not batch:
            return 0
        texts = [f"{item.input.title} {item.input.content}".strip() for item in batch]
        embeddings = await self.embedder.embed_documents(texts)
        faiss_start = self.dense.document_count
        indexed = [
            IndexedDocument(
                prepared=item,
                embedding=tuple(float(value) for value in embeddings[position]),
                faiss_id=faiss_start + position,
                embedding_model=self.embedding_model,
            )
            for position, item in enumerate(batch)
        ]
        await self.repository.persist_batch(indexed)
        candidates = [item.prepared.candidate for item in indexed]
        self.bm25.add(candidates)
        self.dense.add_from_embeddings(candidates, embeddings)
        return len(indexed)

    def _persist_indexes(self) -> str:
        self.bm25.save(self.bm25_index_path)
        if self.dense.document_count:
            self.dense.save(self.faiss_index_path)
        version_source = (
            f"{self.bm25.document_count}:"
            f"{self.dense.document_count}:"
            f"{self.embedding_model}"
        )
        return sha256(version_source.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _prepare(document: DocumentInput) -> PreparedDocument:
        canonical = (
            str(document.canonical_url) if document.canonical_url is not None else ""
        )
        digest_source = "\n".join(
            (
                document.title.strip(),
                document.content.strip(),
                canonical,
            )
        )
        content_hash = sha256(digest_source.encode("utf-8")).hexdigest()
        identity = document.external_id or canonical or content_hash
        document_id = uuid5(NAMESPACE_URL, f"document:{identity}")
        chunk_id = uuid5(NAMESPACE_URL, f"chunk:{identity}:0")
        metadata = {
            **document.metadata,
            "language": document.language,
            "source": document.source,
            "external_id": document.external_id,
        }
        return PreparedDocument(
            input=document,
            content_hash=content_hash,
            candidate=RetrievalCandidate(
                chunk_id=chunk_id,
                document_id=document_id,
                title=document.title,
                content=document.content,
                url=canonical or None,
                metadata=metadata,
            ),
        )


def _batches(
    values: Sequence[PreparedDocument],
    size: int,
) -> list[list[PreparedDocument]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]
