from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.models import RetrievalCandidate
from app.retrieval.bm25 import BM25Retriever
from app.schemas.search import SearchFilters


def document(
    title: str,
    content: str,
    *,
    language: str = "en",
    source: str = "paper",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title=title,
        content=content,
        metadata={"language": language, "source": source},
    )


def test_bm25_ranks_matching_document_first() -> None:
    quantum = document(
        "Quantum retrieval",
        "Hilbert space relevance for search",
    )
    database = document(
        "Database systems",
        "Transaction isolation and recovery",
    )
    retriever = BM25Retriever([database, quantum])

    results = retriever.search_sync(
        "quantum Hilbert search",
        limit=2,
    )

    assert results[0].chunk_id == quantum.chunk_id
    assert results[0].bm25_score > 0


def test_bm25_applies_metadata_filters() -> None:
    english = document("Search", "retrieval engine", language="en")
    french = document("Recherche", "retrieval engine", language="fr")
    retriever = BM25Retriever([english, french])

    results = retriever.search_sync(
        "retrieval",
        limit=10,
        filters=SearchFilters(language="fr"),
    )

    assert [result.chunk_id for result in results] == [french.chunk_id]


def test_bm25_round_trip_persistence(tmp_path: Path) -> None:
    original = document("Persistent search", "BM25 index data")
    path = tmp_path / "bm25.json"
    BM25Retriever([original]).save(path)

    loaded = BM25Retriever.load(path)
    results = loaded.search_sync("persistent", limit=1)

    assert loaded.document_count == 1
    assert results[0].chunk_id == original.chunk_id


def test_bm25_rejects_duplicate_chunk_ids() -> None:
    original = document("One", "content")
    duplicate = RetrievalCandidate(
        chunk_id=original.chunk_id,
        document_id=uuid4(),
        title="Two",
        content="content",
    )
    with pytest.raises(ValueError, match="unique"):
        BM25Retriever([original, duplicate])
