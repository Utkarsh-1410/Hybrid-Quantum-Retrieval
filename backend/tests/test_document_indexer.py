from pathlib import Path
from uuid import UUID

import pytest

from app.indexing.document_indexer import ProductionDocumentIndexer
from app.indexing.repository import InMemoryIndexRepository
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.schemas.documents import DocumentInput
from app.schemas.search import SearchFilters
from tests.retrieval_fakes import FakeEmbedder, InMemoryVectorIndex


def document(
    title: str,
    content: str,
    *,
    external_id: str,
) -> DocumentInput:
    return DocumentInput(
        external_id=external_id,
        title=title,
        content=content,
        source="test",
        metadata={"collection": "unit"},
    )


def make_indexer(
    tmp_path: Path,
) -> tuple[
    ProductionDocumentIndexer,
    InMemoryIndexRepository,
    BM25Retriever,
    DenseRetriever,
]:
    vectors = {
        "Quantum Search state overlap ranking": (1.0, 0.0),
        "Database Systems transaction recovery": (0.0, 1.0),
        "quantum ranking": (1.0, 0.0),
    }
    embedder = FakeEmbedder(vectors)
    repository = InMemoryIndexRepository()
    bm25 = BM25Retriever()
    dense = DenseRetriever(
        embedder,
        index=InMemoryVectorIndex(2),
        documents=[],
    )
    indexer = ProductionDocumentIndexer(
        repository=repository,
        embedder=embedder,
        bm25=bm25,
        dense=dense,
        bm25_index_path=tmp_path / "bm25.json",
        faiss_index_path=tmp_path / "documents.index",
        embedding_model="fake-model",
        batch_size=2,
    )
    return indexer, repository, bm25, dense


@pytest.mark.asyncio
async def test_indexer_batches_and_updates_live_indexes(
    tmp_path: Path,
) -> None:
    indexer, repository, bm25, dense = make_indexer(tmp_path)
    documents = [
        document(
            "Quantum Search",
            "state overlap ranking",
            external_id="q1",
        ),
        document(
            "Database Systems",
            "transaction recovery",
            external_id="d1",
        ),
    ]

    job_id = await indexer.submit(documents)
    snapshot = await indexer.wait_for_job(UUID(job_id))

    assert snapshot.status == "completed"
    assert snapshot.indexed_count == 2
    assert snapshot.skipped_count == 0
    assert len(repository.documents) == 2
    assert bm25.document_count == 2
    assert dense.document_count == 2
    assert (tmp_path / "bm25.json").is_file()

    lexical = bm25.search_sync("quantum", limit=2)
    semantic = await dense.search(
        (1.0, 0.0),
        2,
        SearchFilters(),
    )
    assert lexical[0].title == "Quantum Search"
    assert semantic[0].title == "Quantum Search"


@pytest.mark.asyncio
async def test_indexer_skips_duplicates_across_jobs(tmp_path: Path) -> None:
    indexer, repository, bm25, dense = make_indexer(tmp_path)
    item = document(
        "Quantum Search",
        "state overlap ranking",
        external_id="q1",
    )

    first_id = await indexer.submit([item, item])
    first = await indexer.wait_for_job(UUID(first_id))
    second_id = await indexer.submit([item])
    second = await indexer.wait_for_job(UUID(second_id))

    assert first.indexed_count == 1
    assert first.skipped_count == 1
    assert second.indexed_count == 0
    assert second.skipped_count == 1
    assert len(repository.documents) == 1
    assert bm25.document_count == 1
    assert dense.document_count == 1
