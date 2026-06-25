# Quantum-Inspired Ranking

## Research interpretation

The module follows the vector-space construction used in quantum-inspired
information retrieval. It uses Hilbert-space mathematics on classical
embeddings; it does not require or claim execution on quantum hardware.

Given a nonzero query embedding `q` and document embedding `d`, each is mapped
to a unit state:

```text
|Q> = q / sqrt(sum_i |q_i|^2)
|D> = d / sqrt(sum_i |d_i|^2)
```

The transition amplitude is the complex inner product:

```text
<Q|D> = sum_i conjugate(Q_i) * D_i
```

The Born-rule-inspired relevance probability is:

```text
P(D | Q) = |<Q|D>|^2
```

The value lies in `[0, 1]`. For real normalized embeddings it is squared cosine
similarity. Global phase and sign changes do not alter the probability.

## Contextual model

A centroid can erase context when candidate states cancel. The contextual
ranker instead treats neighboring result states as a classical mixture:

```text
rho_C = sum_j w_j |D_j><D_j| / sum_j w_j
```

For candidate state `|D_i>`, contextual coherence is:

```text
C_i = <D_i|rho_C|D_i>
    = sum_j w_j |<D_j|D_i>|^2 / sum_j w_j
```

By default, the candidate itself is excluded from its context. The weights
`w_j` are derived from nonnegative hybrid retrieval priors. This mixed-state
formulation remains stable when two vectors have opposite signs. When no
neighboring state is available, context falls back to query-state probability.

The final score is configurable:

```text
S_i = alpha * hybrid_i + beta * P(D_i | Q) + gamma * C_i
alpha + beta + gamma = 1
```

Default values are `alpha=0.35`, `beta=0.45`, and `gamma=0.20`.

## Files

- `backend/app/quantum/quantum_ranker.py`: state normalization, transition
  amplitudes, Born probabilities, and pure probability ranking
- `backend/app/quantum/contextual_ranker.py`: mixed-state context and weighted
  application ranking
- `backend/scripts/quantum_ranking_demo.py`: executable example

The former `app.ranking.quantum_ranker` path re-exports these objects for
backward compatibility.

## Example

```python
from app.quantum.contextual_ranker import (
    ContextualRankerConfig,
    QuantumContextualRanker,
)
from app.quantum.quantum_ranker import QuantumRanker

probabilities = QuantumRanker().rank(
    query_vector,
    document_vectors,
    top_k=10,
)

ranker = QuantumContextualRanker(
    ContextualRankerConfig(
        hybrid_weight=0.35,
        quantum_weight=0.45,
        context_weight=0.20,
    )
)
ranked = ranker.rank(
    query_vector,
    hybrid_candidates,
    top_k=10,
)
```

Execute the complete demonstration:

```bash
cd backend
python scripts/quantum_ranking_demo.py
```

Run focused tests:

```bash
pytest tests/test_quantum_ranker.py tests/test_contextual_ranker.py
```

## Limitations

- Squared cosine discards the sign of real-vector similarity.
- Context quality depends on the retrieved candidate pool.
- Weights must be calibrated on validation data, not the final test set.
- Improvements should be reported against cosine and learned reranking
  baselines with per-query statistical analysis.
