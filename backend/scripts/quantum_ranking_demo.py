from uuid import uuid4

from app.domain.models import RetrievalCandidate
from app.quantum.contextual_ranker import (
    ContextualRankerConfig,
    QuantumContextualRanker,
)
from app.quantum.quantum_ranker import QuantumRanker


def document(
    title: str,
    embedding: tuple[float, ...],
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title=title,
        content="Example document",
        embedding=embedding,
    )


def main() -> None:
    query = (1.0, 0.0, 0.0)
    vectors = [
        (0.95, 0.05, 0.0),
        (0.70, 0.30, 0.0),
        (0.0, 0.0, 1.0),
    ]

    print("Pure quantum-inspired probabilities")
    for probability_result in QuantumRanker().rank(query, vectors):
        print(
            f"document={probability_result.index} "
            f"probability={probability_result.probability:.4f}"
        )

    candidates = [
        (document("Quantum retrieval", vectors[0]), 0.75),
        (document("Contextual search", vectors[1]), 0.80),
        (document("Unrelated topic", vectors[2]), 0.90),
    ]
    ranker = QuantumContextualRanker(
        ContextualRankerConfig(
            hybrid_weight=0.35,
            quantum_weight=0.45,
            context_weight=0.20,
        )
    )

    print("\nContextual ranking")
    for ranked_candidate in ranker.rank(query, candidates, top_k=3):
        print(
            f"{ranked_candidate.rank}. {ranked_candidate.candidate.title}: "
            f"prior={ranked_candidate.hybrid_score:.4f}, "
            f"quantum={ranked_candidate.quantum_score:.4f}, "
            f"context={ranked_candidate.context_score:.4f}, "
            f"final={ranked_candidate.final_score:.4f}"
        )


if __name__ == "__main__":
    main()
