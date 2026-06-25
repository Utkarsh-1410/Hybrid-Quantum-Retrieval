from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def paired_bootstrap(
    baseline: Mapping[str, Mapping[str, float]],
    treatment: Mapping[str, Mapping[str, float]],
    *,
    iterations: int = 5000,
    seed: int = 13,
) -> dict[str, dict[str, float]]:
    """Estimate paired metric deltas and 95% bootstrap intervals."""
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    query_ids = sorted(baseline.keys() & treatment.keys())
    if not query_ids:
        raise ValueError("runs do not share evaluated queries")
    metric_names = sorted(baseline[query_ids[0]])
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        len(query_ids),
        size=(iterations, len(query_ids)),
    )
    results: dict[str, dict[str, float]] = {}

    for metric in metric_names:
        deltas = np.asarray(
            [
                treatment[query_id][metric] - baseline[query_id][metric]
                for query_id in query_ids
            ],
            dtype=float,
        )
        bootstrap_means = deltas[sample_indices].mean(axis=1)
        non_positive = float(np.mean(bootstrap_means <= 0))
        non_negative = float(np.mean(bootstrap_means >= 0))
        results[metric] = {
            "mean_delta": round(float(deltas.mean()), 6),
            "ci_95_low": round(float(np.quantile(bootstrap_means, 0.025)), 6),
            "ci_95_high": round(float(np.quantile(bootstrap_means, 0.975)), 6),
            "two_sided_probability": round(
                min(1.0, 2 * min(non_positive, non_negative)), 6
            ),
        }
    return results
