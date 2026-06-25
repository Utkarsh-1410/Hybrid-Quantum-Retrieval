from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import Database
from app.indexing.document_indexer import ProductionDocumentIndexer
from app.indexing.repository import PostgresIndexRepository
from app.quantum.contextual_ranker import (
    QuantumContextualRanker,
    QuantumRankerConfig,
)
from app.rag.generator import GeneratorConfig, LangChainAnswerGenerator
from app.rag.pipeline import RAGPipeline, RAGPipelineConfig
from app.rag.retriever import RAGRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, SentenceTransformerEmbedder
from app.retrieval.hybrid import HybridConfig, HybridRetriever
from app.services.ask import AskService
from app.services.search import SearchService
from app.services.stubs import (
    GroundedPlaceholderGenerator,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    database = Database(settings)

    sentence_embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        batch_size=settings.indexing_batch_size,
    )

    quantum_ranker = QuantumContextualRanker(
        QuantumRankerConfig(
            hybrid_weight=settings.ranker_hybrid_weight,
            quantum_weight=settings.quantum_weight,
            context_weight=settings.context_weight,
        )
    )

    persisted_indexes_exist = (
        settings.bm25_index_path.is_file()
        and settings.faiss_index_path.is_file()
        and settings.faiss_index_path.with_suffix(
            f"{settings.faiss_index_path.suffix}.metadata.json"
        ).is_file()
    )

    print("\n===== INDEX DEBUG =====")
    print("BM25 Path :", settings.bm25_index_path)
    print("FAISS Path:", settings.faiss_index_path)
    print("Indexes Found:", persisted_indexes_exist)

    if persisted_indexes_exist:
        print("Loading persisted indexes...")

        lexical_retriever = BM25Retriever.load(
            settings.bm25_index_path
        )

        dense_retriever = DenseRetriever.load(
            settings.faiss_index_path,
            embedder=sentence_embedder,
        )

        print(
            "BM25 docs:",
            getattr(lexical_retriever, "document_count", "NA"),
        )
        print(
            "Dense docs:",
            getattr(dense_retriever, "document_count", "NA"),
        )
    else:
        print("Indexes not found. Creating empty retrievers.")

        lexical_retriever = BM25Retriever()
        dense_retriever = DenseRetriever(sentence_embedder)

    print("=======================\n")

    hybrid_retriever = HybridRetriever(
        bm25=lexical_retriever,
        dense=dense_retriever,
        embedder=sentence_embedder,
        config=HybridConfig(
            bm25_weight=settings.bm25_weight,
            dense_weight=settings.dense_weight,
        ),
    )

    rag_retriever = RAGRetriever(
        hybrid_retriever=hybrid_retriever,
        quantum_ranker=quantum_ranker,
    )

    search_service = SearchService(
        embedder=sentence_embedder,
        lexical_retriever=lexical_retriever,
        dense_retriever=dense_retriever,
        ranker=quantum_ranker,
        bm25_weight=settings.bm25_weight,
        dense_weight=settings.dense_weight,
    )

    app.state.search_service = search_service

    generator_config = GeneratorConfig(
        max_chars_per_source=settings.rag_max_chars_per_source,
        max_total_context_chars=settings.rag_max_context_chars,
    )

    generator: LangChainAnswerGenerator | GroundedPlaceholderGenerator

    if settings.llm_provider == "openai" and settings.openai_api_key:
        generator = LangChainAnswerGenerator.openai(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            config=generator_config,
        )
    elif settings.llm_provider in {"llama", "ollama"}:
        generator = LangChainAnswerGenerator.llama_ollama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            config=generator_config,
        )
    else:
        generator = GroundedPlaceholderGenerator()

    app.state.ask_service = AskService(
        RAGPipeline(
            retriever=rag_retriever,
            generator=generator,
            config=RAGPipelineConfig(
                minimum_evidence=settings.rag_minimum_evidence
            ),
        )
    )

    indexer = ProductionDocumentIndexer(
        repository=PostgresIndexRepository(database),
        embedder=sentence_embedder,
        bm25=lexical_retriever,
        dense=dense_retriever,
        bm25_index_path=settings.bm25_index_path,
        faiss_index_path=settings.faiss_index_path,
        embedding_model=settings.embedding_model,
        batch_size=settings.indexing_batch_size,
    )

    app.state.indexer = indexer

    try:
        yield
    finally:
        await indexer.close()
        await database.dispose()

def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=("Hybrid BM25, dense, quantum-inspired ranking, and RAG API"),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid4())
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(
            int((perf_counter() - started) * 1000)
        )
        return response

    app.include_router(health_router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
