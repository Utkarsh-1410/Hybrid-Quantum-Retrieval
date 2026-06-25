from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log2

Run = Mapping[str, Mapping[str, float]]
Qrels = Mapping[str, Mapping[str, int]]


def evaluate_run(
    qrels: Qrels,
    run: Run,
    *,
    cutoffs: Sequence[int] = (10, 100),
) -> dict[str, float]:
    return aggregate_query_metrics(evaluate_run_per_query(qrels, run, cutoffs=cutoffs))


def evaluate_run_per_query(
    qrels: Qrels,
    run: Run,
    *,
    cutoffs: Sequence[int] = (10, 100),
) -> dict[str, dict[str, float]]:
    if not qrels:
        raise ValueError("qrels cannot be empty")
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("cutoffs must contain positive integers")

    results: dict[str, dict[str, float]] = {}
    for query_id, judgments in qrels.items():
        ranking = sorted(
            run.get(query_id, {}).items(),
            key=lambda item: (-item[1], item[0]),
        )
        relevant = {
            document_id for document_id, relevance in judgments.items() if relevance > 0
        }
        metrics = {"map": _average_precision(ranking, relevant)}
        for cutoff in cutoffs:
            metrics[f"mrr@{cutoff}"] = _reciprocal_rank(ranking, relevant, cutoff)
            metrics[f"recall@{cutoff}"] = _recall(ranking, relevant, cutoff)
            metrics[f"ndcg@{cutoff}"] = _ndcg(ranking, judgments, cutoff)
        results[query_id] = metrics
    return results


def aggregate_query_metrics(
    per_query: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    if not per_query:
        raise ValueError("per-query metrics cannot be empty")
    metric_names = next(iter(per_query.values())).keys()
    query_count = len(per_query)
    return {
        metric: round(
            sum(values[metric] for values in per_query.values()) / query_count,
            6,
        )
        for metric in metric_names
    }


def _average_precision(
    ranking: list[tuple[str, float]],
    relevant: set[str],
) -> float:
    if not relevant:
        return 0.0
    precision_sum = 0.0
    relevant_seen = 0
    for rank, (document_id, _) in enumerate(ranking, start=1):
        if document_id in relevant:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    return precision_sum / len(relevant)


def _reciprocal_rank(
    ranking: list[tuple[str, float]],
    relevant: set[str],
    cutoff: int,
) -> float:
    for rank, (document_id, _) in enumerate(ranking[:cutoff], start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def _recall(
    ranking: list[tuple[str, float]],
    relevant: set[str],
    cutoff: int,
) -> float:
    if not relevant:
        return 0.0
    retrieved = {
        document_id for document_id, _ in ranking[:cutoff] if document_id in relevant
    }
    return len(retrieved) / len(relevant)


def _ndcg(
    ranking: list[tuple[str, float]],
    judgments: Mapping[str, int],
    cutoff: int,
) -> float:
    gains = [judgments.get(document_id, 0) for document_id, _ in ranking[:cutoff]]
    ideal = sorted(judgments.values(), reverse=True)[:cutoff]
    ideal_dcg = _dcg(ideal)
    return 0.0 if ideal_dcg == 0 else _dcg(gains) / ideal_dcg


def _dcg(relevances: Sequence[int]) -> float:
    return float(
        sum(
            ((2**relevance) - 1) / log2(rank + 1)
            for rank, relevance in enumerate(relevances, start=1)
        )
    )
