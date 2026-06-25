from math import sqrt

import pytest

from app.quantum.quantum_ranker import (
    QuantumRanker,
    normalize_state,
    quantum_similarity,
    transition_amplitude,
)


def test_normalize_state_has_unit_norm() -> None:
    assert normalize_state((3.0, 4.0)) == pytest.approx((0.6, 0.8))


def test_quantum_similarity_is_born_probability() -> None:
    assert quantum_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert quantum_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_transition_amplitude_uses_complex_conjugation() -> None:
    scale = 1 / sqrt(2)
    query = (scale, 1j * scale)
    orthogonal = (scale, -1j * scale)

    assert transition_amplitude(query, orthogonal) == pytest.approx(0j)


def test_probability_is_invariant_to_global_phase() -> None:
    query = (1.0, 2.0)
    phased_document = (1j, 2j)

    assert quantum_similarity(query, phased_document) == pytest.approx(1.0)


def test_quantum_ranker_orders_documents_by_probability() -> None:
    ranked = QuantumRanker().rank(
        (1.0, 0.0),
        [(0.0, 1.0), (0.7, 0.7), (1.0, 0.0)],
        top_k=2,
    )

    assert [item.index for item in ranked] == [2, 1]
    assert ranked[0].probability == pytest.approx(1.0)
    assert ranked[1].probability == pytest.approx(0.5)


def test_invalid_quantum_states_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        normalize_state((0.0, 0.0))
    with pytest.raises(ValueError, match="equal dimensions"):
        quantum_similarity((1.0,), (1.0, 0.0))
