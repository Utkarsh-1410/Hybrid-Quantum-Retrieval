from time import perf_counter
from uuid import uuid4

from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RAGEvidence
from app.schemas.ask import AskRequest, AskResponse, Citation
from app.schemas.search import ScoreExplanation, SearchResult


class AskService:
    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    async def ask(self, request: AskRequest) -> AskResponse:
        started = perf_counter()
        result = await self.pipeline.ask(
            request.query,
            top_k=request.top_k,
            filters=request.filters,
        )
        return AskResponse(
            request_id=uuid4(),
            answer=result.answer,
            citations=[self._citation(evidence) for evidence in result.citations],
            sources=[self._source(evidence) for evidence in result.evidence],
            confidence=result.confidence,
            latency_ms=int((perf_counter() - started) * 1000),
        )

    @staticmethod
    def _citation(evidence: RAGEvidence) -> Citation:
        candidate = evidence.candidate
        return Citation(
            citation_id=evidence.citation_id,
            rank=evidence.rank,
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
            title=candidate.title,
            url=candidate.url,
            excerpt=candidate.content[:500],
            score=evidence.final_score,
        )

    @staticmethod
    def _source(evidence: RAGEvidence) -> SearchResult:
        candidate = evidence.candidate
        return SearchResult(
            rank=evidence.rank,
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
            title=candidate.title,
            snippet=candidate.content[:500],
            url=candidate.url,
            metadata=candidate.metadata,
            scores=ScoreExplanation(
                bm25=candidate.bm25_score,
                dense=candidate.dense_score,
                hybrid=evidence.hybrid_score,
                quantum=evidence.quantum_score,
                context=evidence.context_score,
                final=evidence.final_score,
                formula=("configured hybrid + quantum + contextual RAG score"),
            ),
        )
