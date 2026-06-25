from uuid import uuid4

import pytest

from app.domain.models import RetrievalCandidate
from app.quantum.contextual_ranker import (
    ContextualRankerConfig,
    QuantumContextualRanker,
)


def candidate(
    title: str,
    embedding: tuple[float, ...],
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title=title,
        content="Test content",
        embedding=embedding,
    )


def test_contextual_ranker_prefers_query_aligned_document() -> None:
    aligned = candidate("aligned", (1.0, 0.0))
    orthogonal = candidate("orthogonal", (0.0, 1.0))

    ranked = QuantumContextualRanker().rank(
        (1.0, 0.0),
        [(orthogonal, 0.5), (aligned, 0.5)],
        top_k=2,
    )

    assert ranked[0].candidate.chunk_id == aligned.chunk_id
    assert ranked[0].quantum_score == pytest.approx(1.0)


def test_mixed_context_rewards_candidate_coherence() -> None:
    first = candidate("cluster-one", (1.0, 0.0))
    second = candidate("cluster-two", (0.9, 0.1))
    outlier = candidate("outlier", (0.0, 1.0))
    ranker = QuantumContextualRanker(
        ContextualRankerConfig(
            hybrid_weight=0.0,
            quantum_weight=0.0,
            context_weight=1.0,
        )
    )

    ranked = ranker.rank(
        (1.0, 0.0),
        [(first, 0.5), (second, 0.5), (outlier, 0.5)],
        top_k=3,
    )

    assert ranked[-1].candidate.chunk_id == outlier.chunk_id
    assert ranked[0].context_score > ranked[-1].context_score


def test_context_uses_density_mixture_without_vector_cancellation() -> None:
    positive = candidate("positive", (1.0, 0.0))
    negative = candidate("negative", (-1.0, 0.0))
    ranker = QuantumContextualRanker(
        ContextualRankerConfig(
            hybrid_weight=0.0,
            quantum_weight=0.0,
            context_weight=1.0,
        )
    )

    ranked = ranker.rank(
        (1.0, 0.0),
        [(positive, 0.5), (negative, 0.5)],
        top_k=2,
    )

    assert all(item.context_score == pytest.approx(1.0) for item in ranked)


def test_configurable_weights_control_final_score() -> None:
    document = candidate("document", (1.0, 0.0))
    ranker = QuantumContextualRanker(
        ContextualRankerConfig(
            hybrid_weight=0.25,
            quantum_weight=0.75,
            context_weight=0.0,
        )
    )

    result = ranker.rank(
        (1.0, 0.0),
        [(document, 0.4)],
        top_k=1,
    )[0]

    assert result.final_score == pytest.approx(0.85)


def test_candidates_without_embeddings_fall_back_to_prior() -> None:
    document = candidate("missing", ())

    result = QuantumContextualRanker().rank(
        (1.0, 0.0),
        [(document, 0.8)],
        top_k=1,
    )[0]

    assert result.final_score == pytest.approx(0.8)
    assert result.quantum_score == 0.0


def test_invalid_component_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ContextualRankerConfig(
            hybrid_weight=0.5,
            quantum_weight=0.5,
            context_weight=0.5,
        )


def test_legacy_ranking_import_remains_available() -> None:
    from app.ranking.quantum_ranker import QuantumContextualRanker as Legacy

    assert Legacy is QuantumContextualRanker
