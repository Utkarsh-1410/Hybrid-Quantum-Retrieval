import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.evaluation.runner import BenchmarkConfig, run_scifact_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BM25, dense, hybrid, and quantum-contextual retrieval "
            "on the BEIR SciFact benchmark."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/benchmarks"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/scifact.json"),
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit evaluated queries for a smoke run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_scifact_benchmark(
        BenchmarkConfig(
            cache_dir=args.cache_dir,
            output_path=args.output,
            model_name=args.model,
            candidate_pool=args.candidate_pool,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_queries=args.max_queries,
        )
    )
    summary = {
        "output": str(args.output),
        "metrics": report["metrics"],
        "quantum_vs_hybrid_delta": report["quantum_vs_hybrid_delta"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
