import asyncio
from pathlib import Path
from uuid import uuid4

from app.domain.models import RetrievalCandidate
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, SentenceTransformerEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.schemas.search import SearchFilters


def sample_documents() -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=uuid4(),
            title="Hybrid information retrieval",
            content=(
                "BM25 lexical matching can be combined with dense semantic retrieval."
            ),
            metadata={"language": "en", "source": "demo"},
        ),
        RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=uuid4(),
            title="Quantum-inspired ranking",
            content=(
                "A normalized query and document state can be compared using "
                "squared inner-product overlap."
            ),
            metadata={"language": "en", "source": "demo"},
        ),
        RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=uuid4(),
            title="Database indexing",
            content="B-tree indexes accelerate exact database lookups.",
            metadata={"language": "en", "source": "demo"},
        ),
    ]


async def main() -> None:
    documents = sample_documents()
    embedder = SentenceTransformerEmbedder()
    bm25 = BM25Retriever(documents)
    dense = DenseRetriever(embedder)
    await dense.build(documents)

    index_directory = Path("../data/faiss")
    bm25.save(index_directory / "demo-bm25.json")
    dense.save(index_directory / "demo-dense.index")

    retriever = HybridRetriever(
        bm25=bm25,
        dense=dense,
        embedder=embedder,
    )
    results = await retriever.search(
        "semantic quantum search",
        limit=3,
        filters=SearchFilters(language="en"),
    )
    for rank, (candidate, score) in enumerate(results, start=1):
        print(f"{rank}. {candidate.title} ({score:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
