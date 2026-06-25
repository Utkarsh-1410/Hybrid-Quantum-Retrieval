import argparse
import asyncio
from datetime import date
from pathlib import Path

from app.datasets.arxiv import ArxivFilter, iter_arxiv_candidates
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, SentenceTransformerEmbedder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build local BM25 and FAISS indexes from the arXiv metadata snapshot."
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/faiss/arxiv"),
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=10_000,
        help=(
            "Maximum selected papers. Increase deliberately; embedding the "
            "full multi-million-paper snapshot is expensive."
        ),
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Filter by category or category family, e.g. cs or cs.IR.",
    )
    parser.add_argument("--updated-from", type=date.fromisoformat)
    parser.add_argument("--updated-to", type=date.fromisoformat)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-invalid", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> None:
    filters = ArxivFilter(
        categories=tuple(args.category),
        updated_from=args.updated_from,
        updated_to=args.updated_to,
    )
    documents = list(
        iter_arxiv_candidates(
            args.dataset,
            filters=filters,
            limit=args.max_documents,
            skip_invalid=args.skip_invalid,
        )
    )
    if not documents:
        raise ValueError("No arXiv papers matched the requested filters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bm25 = BM25Retriever(documents)
    bm25.save(args.output_dir / "bm25.json")

    embedder = SentenceTransformerEmbedder(
        args.model,
        batch_size=args.batch_size,
    )
    dense = DenseRetriever(embedder)
    await dense.build(documents)
    dense.save(args.output_dir / "documents.index")
    print(f"Indexed {len(documents)} arXiv papers")
    print(f"BM25: {args.output_dir / 'bm25.json'}")
    print(f"FAISS: {args.output_dir / 'documents.index'}")


def main() -> int:
    args = build_parser().parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
