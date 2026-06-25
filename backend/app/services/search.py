import asyncio
from time import perf_counter
from uuid import uuid4

from app.quantum.contextual_ranker import QuantumContextualRanker
from app.retrieval.hybrid import fuse_candidates
from app.schemas.search import (
    ScoreExplanation,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.protocols import DenseRetriever, Embedder, LexicalRetriever


class SearchService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        lexical_retriever: LexicalRetriever,
        dense_retriever: DenseRetriever,
        ranker: QuantumContextualRanker,
        bm25_weight: float,
        dense_weight: float,
    ) -> None:
        self.embedder = embedder
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.ranker = ranker
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = perf_counter()

        lexical_count = getattr(
            self.lexical_retriever,
            "document_count",
            None,
        )

        dense_count = getattr(
            self.dense_retriever,
            "document_count",
            None,
        )

        print("\n===== SEARCH DEBUG =====")
        print("Query:", request.query)
        print("BM25 Documents:", lexical_count)
        print("Dense Documents:", dense_count)

        if lexical_count == 0 and dense_count == 0:
            print("No indexed documents available")
            print("========================\n")

            return SearchResponse(
                request_id=uuid4(),
                query=request.query,
                total=0,
                latency_ms=int((perf_counter() - started) * 1000),
                results=[],
            )

        query_embedding = await self.embedder.embed_query(
            request.query
        )

        candidate_limit = max(
            request.top_k * 5,
            25,
        )

        lexical, dense = await asyncio.gather(
            self.lexical_retriever.search(
                request.query,
                candidate_limit,
                request.filters,
            ),
            self.dense_retriever.search(
                query_embedding,
                candidate_limit,
                request.filters,
            ),
        )

        print("BM25 Returned:", len(lexical))
        print("Dense Returned:", len(dense))

        if lexical:
            print("Top BM25 Title:", lexical[0].title)

        if dense:
            print("Top Dense Title:", dense[0].title)

        fused = fuse_candidates(
            lexical,
            dense,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )

        print("Fused Candidates:", len(fused))

        ranked = self.ranker.rank(
            query_embedding,
            fused,
            top_k=request.top_k,
        )

        print("Ranked Results:", len(ranked))
        print("========================\n")

        config = self.ranker.config

        formula = (
            f"{config.hybrid_weight:.2f}*hybrid + "
            f"{config.quantum_weight:.2f}*quantum + "
            f"{config.context_weight:.2f}*context"
        )

        results = [
            SearchResult(
                rank=item.rank,
                document_id=item.candidate.document_id,
                chunk_id=item.candidate.chunk_id,
                title=item.candidate.title,
                snippet=item.candidate.content[:500],
                url=item.candidate.url,
                metadata=item.candidate.metadata,
                scores=ScoreExplanation(
                    bm25=item.candidate.bm25_score,
                    dense=item.candidate.dense_score,
                    hybrid=item.hybrid_score,
                    quantum=item.quantum_score,
                    context=item.context_score,
                    final=item.final_score,
                    formula=formula,
                ),
            )
            for item in ranked
        ]

        return SearchResponse(
            request_id=uuid4(),
            query=request.query,
            total=len(results),
            latency_ms=int((perf_counter() - started) * 1000),
            results=results,
        )