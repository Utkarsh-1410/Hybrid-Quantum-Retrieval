from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.domain.models import RetrievalCandidate
from app.retrieval.dense import (
    DenseRetriever,
    FaissFlatIPIndex,
    SentenceTransformerEmbedder,
    normalize_rows,
)
from app.schemas.search import SearchFilters
from tests.retrieval_fakes import FakeEmbedder, InMemoryVectorIndex


def document(
    title: str,
    content: str,
    *,
    source: str,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title=title,
        content=content,
        metadata={"language": "en", "source": source},
    )


@pytest.mark.asyncio
async def test_dense_retriever_returns_cosine_nearest_document() -> None:
    quantum = document("Quantum", "ranking", source="paper")
    database = document("Database", "transactions", source="book")
    embedder = FakeEmbedder(
        {
            "Quantum ranking": (1.0, 0.0),
            "Database transactions": (0.0, 1.0),
            "quantum query": (0.9, 0.1),
        }
    )
    retriever = DenseRetriever(embedder)
    embeddings = await embedder.embed_documents(
        ["Quantum ranking", "Database transactions"]
    )
    retriever.build_from_embeddings(
        [quantum, database],
        embeddings,
        index=InMemoryVectorIndex(2),
    )

    query = await embedder.embed_query("quantum query")
    results = await retriever.search(query, 2, SearchFilters())

    assert results[0].chunk_id == quantum.chunk_id
    assert results[0].dense_score > results[1].dense_score
    assert np.linalg.norm(results[0].embedding) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_dense_retriever_filters_after_vector_search() -> None:
    paper = document("Quantum", "ranking", source="paper")
    book = document("Search", "systems", source="book")
    embedder = FakeEmbedder({"query": (1.0, 0.0)})
    retriever = DenseRetriever(embedder)
    retriever.build_from_embeddings(
        [paper, book],
        np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        index=InMemoryVectorIndex(2),
    )

    results = await retriever.search(
        (1.0, 0.0),
        1,
        SearchFilters(source="book"),
    )

    assert [result.chunk_id for result in results] == [book.chunk_id]


def test_normalize_rows_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        normalize_rows(np.asarray([[0.0, 0.0]], dtype=np.float32))


@pytest.mark.asyncio
async def test_sentence_transformer_adapter_normalizes_model_output() -> None:
    class FakeModel:
        def encode(self, texts: object, **kwargs: object) -> np.ndarray:
            assert kwargs["normalize_embeddings"] is True
            return np.asarray([[3.0, 4.0]], dtype=np.float32)

    embedder = SentenceTransformerEmbedder(model=FakeModel())

    embedding = await embedder.embed_query("query")

    assert embedding == pytest.approx((0.6, 0.8))


def test_faiss_index_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    path = tmp_path / "vectors.index"
    index = FaissFlatIPIndex.create(2)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    index.save(path)

    loaded = FaissFlatIPIndex.load(path)
    scores, positions = loaded.search(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        2,
    )

    assert loaded.size == 2
    assert positions[0, 0] == 0
    assert scores[0, 0] == pytest.approx(1.0)
