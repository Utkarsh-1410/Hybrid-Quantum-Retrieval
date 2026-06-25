import argparse
import asyncio
import os
from collections.abc import Sequence
from uuid import uuid4

from app.domain.models import RetrievalCandidate
from app.quantum.contextual_ranker import QuantumContextualRanker
from app.rag.generator import LangChainAnswerGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RAGRetriever
from app.retrieval.hybrid import HybridRetrievalResult
from app.schemas.search import SearchFilters


class DemoHybridRetriever:
    async def retrieve(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> HybridRetrievalResult:
        candidates = [
            RetrievalCandidate(
                chunk_id=uuid4(),
                document_id=uuid4(),
                title="Quantum-inspired relevance",
                content=(
                    "Normalized query and document vectors are treated as "
                    "states. Their relevance is the squared inner product."
                ),
                embedding=(1.0, 0.0),
                bm25_score=0.7,
                dense_score=0.9,
            ),
            RetrievalCandidate(
                chunk_id=uuid4(),
                document_id=uuid4(),
                title="Hybrid retrieval",
                content=(
                    "The system combines normalized BM25 and dense scores "
                    "before contextual re-ranking."
                ),
                embedding=(0.8, 0.2),
                bm25_score=0.9,
                dense_score=0.7,
            ),
        ]
        return HybridRetrievalResult(
            query_embedding=(1.0, 0.0),
            candidates=[(candidate, 0.8) for candidate in candidates],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the quantum-ranked RAG pipeline demonstration."
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "ollama"),
        default=os.getenv("LLM_PROVIDER", "ollama"),
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    provider = args.provider
    if provider == "openai":
        generator = LangChainAnswerGenerator.openai(
            model=args.model or "gpt-4.1-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    else:
        generator = LangChainAnswerGenerator.llama_ollama(
            model=args.model or "llama3.1",
            base_url=args.ollama_base_url,
        )

    pipeline = RAGPipeline(
        retriever=RAGRetriever(
            hybrid_retriever=DemoHybridRetriever(),
            quantum_ranker=QuantumContextualRanker(),
        ),
        generator=generator,
    )
    result = await pipeline.ask(
        "How does quantum-inspired hybrid ranking work?",
        top_k=2,
    )
    print(result.answer)
    print(f"Confidence: {result.confidence:.4f}")
    print("Citations:", [item.citation_id for item in result.citations])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
