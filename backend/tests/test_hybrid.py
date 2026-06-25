from uuid import uuid4

import numpy as np
import pytest

from app.domain.models import RetrievalCandidate
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever, fuse_candidates
from app.schemas.search import SearchFilters
from tests.retrieval_fakes import FakeEmbedder, InMemoryVectorIndex


def candidate(*, bm25: float = 0.0, dense: float = 0.0) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title="Test",
        content="Test content",
        bm25_score=bm25,
        dense_score=dense,
    )


def test_fusion_applies_requested_weights() -> None:
    lexical_best = candidate(bm25=10)
    lexical_low = candidate(bm25=1)
    dense_best = candidate(dense=0.9)
    dense_low = candidate(dense=0.1)

    fused = fuse_candidates(
        [lexical_best, lexical_low],
        [dense_best, dense_low],
    )
    scores = {item.chunk_id: score for item, score in fused}

    assert scores[lexical_best.chunk_id] == pytest.approx(0.4)
    assert scores[dense_best.chunk_id] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_hybrid_retriever_combines_bm25_and_dense_search() -> None:
    lexical_match = RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title="Quantum search",
        content="exact lexical terms",
    )
    semantic_match = RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title="Hilbert retrieval",
        content="semantic relevance",
    )
    texts = {
        "Quantum search exact lexical terms": (0.5, 0.5),
        "Hilbert retrieval semantic relevance": (1.0, 0.0),
        "quantum search": (1.0, 0.0),
    }
    embedder = FakeEmbedder(texts)
    dense = DenseRetriever(embedder)
    dense.build_from_embeddings(
        [lexical_match, semantic_match],
        np.asarray(
            [
                texts["Quantum search exact lexical terms"],
                texts["Hilbert retrieval semantic relevance"],
            ],
            dtype=np.float32,
        ),
        index=InMemoryVectorIndex(2),
    )
    hybrid = HybridRetriever(
        bm25=BM25Retriever([lexical_match, semantic_match]),
        dense=dense,
        embedder=embedder,
    )

    results = await hybrid.search(
        "quantum search",
        limit=2,
        filters=SearchFilters(),
    )

    assert len(results) == 2
    assert all(0 <= score <= 1 for _, score in results)
    assert {item.chunk_id for item, _ in results} == {
        lexical_match.chunk_id,
        semantic_match.chunk_id,
    }
