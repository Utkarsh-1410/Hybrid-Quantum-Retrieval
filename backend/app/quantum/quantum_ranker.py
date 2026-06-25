from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt

Scalar = float | complex
Vector = Sequence[Scalar]
QuantumState = tuple[complex, ...]


def normalize_state(vector: Vector) -> QuantumState:
    """Map a nonzero vector to a unit state in a complex Hilbert space."""
    if not vector:
        raise ValueError("state vector cannot be empty")
    state = tuple(complex(value) for value in vector)
    if any(not isfinite(value.real) or not isfinite(value.imag) for value in state):
        raise ValueError("state vector must contain finite values")
    norm_squared = fsum(abs(value) ** 2 for value in state)
    if norm_squared == 0:
        raise ValueError("zero vector cannot represent a quantum state")
    norm = sqrt(norm_squared)
    return tuple(value / norm for value in state)


def transition_amplitude(
    query_state: Vector,
    document_state: Vector,
) -> complex:
    """Compute the conjugate inner product <Q|D>."""
    if len(query_state) != len(document_state):
        raise ValueError("query and document states must have equal dimensions")
    query = normalize_state(query_state)
    document = normalize_state(document_state)
    return sum(
        query_value.conjugate() * document_value
        for query_value, document_value in zip(
            query,
            document,
            strict=True,
        )
    )


def quantum_similarity(
    query_state: Vector,
    document_state: Vector,
) -> float:
    """Compute the Born-rule-inspired probability |<Q|D>| squared."""
    amplitude = transition_amplitude(query_state, document_state)
    probability = abs(amplitude) ** 2
    return min(1.0, max(0.0, float(probability)))


@dataclass(frozen=True, slots=True)
class QuantumRankedState:
    index: int
    probability: float
    amplitude: complex


class QuantumRanker:
    """Rank document vectors by their transition probability from a query."""

    def score(
        self,
        query_vector: Vector,
        document_vector: Vector,
    ) -> QuantumRankedState:
        amplitude = transition_amplitude(query_vector, document_vector)
        return QuantumRankedState(
            index=0,
            probability=min(1.0, max(0.0, float(abs(amplitude) ** 2))),
            amplitude=amplitude,
        )

    def rank(
        self,
        query_vector: Vector,
        document_vectors: Sequence[Vector],
        *,
        top_k: int | None = None,
    ) -> list[QuantumRankedState]:
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        if not document_vectors:
            return []

        query = normalize_state(query_vector)
        ranked: list[QuantumRankedState] = []
        for index, document_vector in enumerate(document_vectors):
            document = normalize_state(document_vector)
            if len(query) != len(document):
                raise ValueError("query and document states must have equal dimensions")
            amplitude = sum(
                query_value.conjugate() * document_value
                for query_value, document_value in zip(
                    query,
                    document,
                    strict=True,
                )
            )
            ranked.append(
                QuantumRankedState(
                    index=index,
                    probability=min(
                        1.0,
                        max(0.0, float(abs(amplitude) ** 2)),
                    ),
                    amplitude=amplitude,
                )
            )

        ranked.sort(key=lambda item: (-item.probability, item.index))
        return ranked if top_k is None else ranked[:top_k]
