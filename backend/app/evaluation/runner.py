from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray

from app.domain.models import RetrievalCandidate
from app.evaluation.dataset import (
    ensure_scifact_dataset,
    load_beir_dataset,
)
from app.evaluation.fusion import fuse_hits
from app.evaluation.metrics import (
    aggregate_query_metrics,
    evaluate_run_per_query,
)
from app.evaluation.models import BenchmarkDataset, FusedHit, RetrievalHit
from app.evaluation.retrievers import BM25BenchmarkIndex, DenseBenchmarkIndex
from app.evaluation.statistics import paired_bootstrap
from app.quantum.contextual_ranker import (
    QuantumContextualRanker,
    QuantumRankerConfig,
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    cache_dir: Path
    output_path: Path
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    split: str = "test"
    candidate_pool: int = 100
    top_k: int = 100
    batch_size: int = 64
    max_queries: int | None = None
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    ranker_hybrid_weight: float = 0.35
    quantum_weight: float = 0.45
    context_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.candidate_pool < self.top_k:
            raise ValueError("candidate_pool must be at least top_k")
        if self.top_k < 1 or self.batch_size < 1:
            raise ValueError("top_k and batch_size must be positive")
        if self.max_queries is not None and self.max_queries < 1:
            raise ValueError("max_queries must be positive")


def run_scifact_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    dataset_dir = ensure_scifact_dataset(config.cache_dir)
    dataset = load_beir_dataset(
        dataset_dir,
        name="scifact",
        split=config.split,
    )
    report = run_benchmark(dataset, config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def run_benchmark(
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> dict[str, object]:
    documents = list(dataset.corpus.values())
    bm25 = BM25BenchmarkIndex(documents)
    dense = DenseBenchmarkIndex(
        documents,
        model_name=config.model_name,
        batch_size=config.batch_size,
    )
    ranker = QuantumContextualRanker(
        QuantumRankerConfig(
            hybrid_weight=config.ranker_hybrid_weight,
            quantum_weight=config.quantum_weight,
            context_weight=config.context_weight,
        )
    )
    runs: dict[str, dict[str, dict[str, float]]] = {
        "bm25": {},
        "dense": {},
        "hybrid": {},
        "quantum_contextual": {},
    }
    query_items = list(dataset.queries.items())
    if config.max_queries is not None:
        query_items = query_items[: config.max_queries]

    for query_id, query in query_items:
        query_embedding = dense.encode_query(query)
        lexical_hits = bm25.search(query, top_k=config.candidate_pool)
        dense_hits = dense.search_embedding(
            query_embedding,
            top_k=config.candidate_pool,
        )
        fused_hits = fuse_hits(
            lexical_hits,
            dense_hits,
            bm25_weight=config.bm25_weight,
            dense_weight=config.dense_weight,
            top_k=config.top_k,
        )
        quantum_hits = _quantum_rerank(
            query_embedding=query_embedding,
            fused_hits=fused_hits,
            dataset=dataset,
            dense=dense,
            ranker=ranker,
            top_k=config.top_k,
        )
        runs["bm25"][query_id] = _score_map(lexical_hits[: config.top_k])
        runs["dense"][query_id] = _score_map(dense_hits[: config.top_k])
        runs["hybrid"][query_id] = {
            hit.document_id: hit.hybrid_score for hit in fused_hits
        }
        runs["quantum_contextual"][query_id] = quantum_hits

    evaluated_qrels = {
        query_id: dataset.qrels[query_id]
        for query_id, _ in query_items
        if query_id in dataset.qrels
    }
    per_query_metrics = {
        name: evaluate_run_per_query(evaluated_qrels, run) for name, run in runs.items()
    }
    metrics = {
        name: aggregate_query_metrics(values)
        for name, values in per_query_metrics.items()
    }
    hybrid_metrics = metrics["hybrid"]
    quantum_metrics = metrics["quantum_contextual"]
    deltas = {
        metric: round(quantum_metrics[metric] - value, 6)
        for metric, value in hybrid_metrics.items()
    }
    return {
        "benchmark": dataset.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "documents": len(dataset.corpus),
            "queries": len(query_items),
            "judged_queries": len(evaluated_qrels),
            "split": config.split,
        },
        "configuration": {
            **asdict(config),
            "cache_dir": str(config.cache_dir),
            "output_path": str(config.output_path),
        },
        "metrics": metrics,
        "quantum_vs_hybrid_delta": deltas,
        "paired_bootstrap": paired_bootstrap(
            per_query_metrics["hybrid"],
            per_query_metrics["quantum_contextual"],
        ),
    }


def _quantum_rerank(
    *,
    query_embedding: NDArray[np.float32],
    fused_hits: list[FusedHit],
    dataset: BenchmarkDataset,
    dense: DenseBenchmarkIndex,
    ranker: QuantumContextualRanker,
    top_k: int,
) -> dict[str, float]:
    candidates: list[tuple[RetrievalCandidate, float]] = []
    for hit in fused_hits:
        document = dataset.corpus[hit.document_id]
        candidate = RetrievalCandidate(
            chunk_id=uuid5(NAMESPACE_URL, f"{dataset.name}:chunk:{hit.document_id}"),
            document_id=uuid5(
                NAMESPACE_URL, f"{dataset.name}:document:{hit.document_id}"
            ),
            title=document.title,
            content=document.text,
            embedding=tuple(
                float(value) for value in dense.embedding_for(hit.document_id)
            ),
            bm25_score=hit.bm25_score,
            dense_score=hit.dense_score,
            metadata={"benchmark_document_id": hit.document_id},
        )
        candidates.append((candidate, hit.hybrid_score))

    query_state = tuple(float(value) for value in query_embedding)
    ranked = ranker.rank(query_state, candidates, top_k=top_k)
    return {
        str(item.candidate.metadata["benchmark_document_id"]): item.final_score
        for item in ranked
    }


def _score_map(hits: list[RetrievalHit]) -> dict[str, float]:
    return {hit.document_id: hit.score for hit in hits}
