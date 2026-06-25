from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite

from app.domain.models import RankedCandidate, RetrievalCandidate
from app.quantum.quantum_ranker import Vector, quantum_similarity


@dataclass(frozen=True, slots=True)
class ContextualRankerConfig:
    """Weights for prior relevance, state overlap, and contextual coherence."""

    hybrid_weight: float = 0.35
    quantum_weight: float = 0.45
    context_weight: float = 0.20
    query_context_weight: float = 0.0
    context_prior_power: float = 1.0
    exclude_self_from_context: bool = True

    def __post_init__(self) -> None:
        component_weights = (
            self.hybrid_weight,
            self.quantum_weight,
            self.context_weight,
        )
        if any(not isfinite(weight) or weight < 0 for weight in component_weights):
            raise ValueError("ranking weights must be finite and nonnegative")
        if abs(fsum(component_weights) - 1.0) > 1e-9:
            raise ValueError("ranking weights must sum to 1")
        if not 0 <= self.query_context_weight <= 1:
            raise ValueError("query_context_weight must be between 0 and 1")
        if not isfinite(self.context_prior_power) or self.context_prior_power < 0:
            raise ValueError("context_prior_power must be finite and nonnegative")


class QuantumContextualRanker:
    """Blend hybrid priors with pure-state and mixed-context probabilities."""

    def __init__(
        self,
        config: ContextualRankerConfig | None = None,
    ) -> None:
        self.config = config or ContextualRankerConfig()

    def rank(
        self,
        query_embedding: Vector,
        candidates: list[tuple[RetrievalCandidate, float]],
        *,
        top_k: int,
    ) -> list[RankedCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []

        valid_indices = [
            index
            for index, (candidate, _) in enumerate(candidates)
            if candidate.embedding and len(candidate.embedding) == len(query_embedding)
        ]
        if not valid_indices:
            return self._rank_without_embeddings(candidates, top_k)

        priors = self._context_priors(candidates, valid_indices)
        scored: list[tuple[RetrievalCandidate, float, float, float, float]] = []
        for index, (candidate, hybrid_score) in enumerate(candidates):
            if index not in valid_indices:
                scored.append((candidate, hybrid_score, 0.0, 0.0, hybrid_score))
                continue

            quantum_score = quantum_similarity(
                query_embedding,
                candidate.embedding,
            )
            neighborhood_score = self._mixed_context_probability(
                candidate_index=index,
                candidates=candidates,
                valid_indices=valid_indices,
                priors=priors,
                fallback_probability=quantum_score,
            )
            context_score = (
                self.config.query_context_weight * quantum_score
                + (1 - self.config.query_context_weight) * neighborhood_score
            )
            final_score = (
                self.config.hybrid_weight * hybrid_score
                + self.config.quantum_weight * quantum_score
                + self.config.context_weight * context_score
            )
            scored.append(
                (
                    candidate,
                    hybrid_score,
                    quantum_score,
                    context_score,
                    final_score,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[4],
                str(item[0].chunk_id),
            )
        )
        return [
            RankedCandidate(
                candidate=candidate,
                rank=rank,
                hybrid_score=hybrid_score,
                quantum_score=quantum_score,
                context_score=context_score,
                final_score=final_score,
            )
            for rank, (
                candidate,
                hybrid_score,
                quantum_score,
                context_score,
                final_score,
            ) in enumerate(scored[:top_k], start=1)
        ]

    def _mixed_context_probability(
        self,
        *,
        candidate_index: int,
        candidates: list[tuple[RetrievalCandidate, float]],
        valid_indices: list[int],
        priors: dict[int, float],
        fallback_probability: float,
    ) -> float:
        context_indices = [
            index
            for index in valid_indices
            if not (self.config.exclude_self_from_context and index == candidate_index)
        ]
        candidate_state = candidates[candidate_index][0].embedding
        if not context_indices:
            return fallback_probability

        context_weights = [priors[index] for index in context_indices]
        total_weight = fsum(context_weights)
        if total_weight == 0:
            context_weights = [1.0] * len(context_indices)
            total_weight = float(len(context_indices))

        return (
            fsum(
                weight
                * quantum_similarity(
                    candidates[index][0].embedding,
                    candidate_state,
                )
                for index, weight in zip(
                    context_indices,
                    context_weights,
                    strict=True,
                )
            )
            / total_weight
        )

    def _context_priors(
        self,
        candidates: list[tuple[RetrievalCandidate, float]],
        valid_indices: list[int],
    ) -> dict[int, float]:
        priors: dict[int, float] = {}
        for index in valid_indices:
            hybrid_score = candidates[index][1]
            if not isfinite(hybrid_score) or hybrid_score < 0:
                raise ValueError("hybrid scores must be finite and nonnegative")
            priors[index] = hybrid_score**self.config.context_prior_power
        return priors

    @staticmethod
    def _rank_without_embeddings(
        candidates: list[tuple[RetrievalCandidate, float]],
        top_k: int,
    ) -> list[RankedCandidate]:
        ordered = sorted(
            candidates,
            key=lambda item: (-item[1], str(item[0].chunk_id)),
        )
        return [
            RankedCandidate(
                candidate=candidate,
                rank=rank,
                hybrid_score=hybrid_score,
                quantum_score=0.0,
                context_score=0.0,
                final_score=hybrid_score,
            )
            for rank, (candidate, hybrid_score) in enumerate(
                ordered[:top_k],
                start=1,
            )
        ]


QuantumRankerConfig = ContextualRankerConfig
