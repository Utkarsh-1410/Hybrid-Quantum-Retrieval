from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.evaluation.models import BenchmarkDocument, RetrievalHit

TOKEN_PATTERN = re.compile(r"\b[\w-]+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class BM25BenchmarkIndex:
    def __init__(self, documents: Sequence[BenchmarkDocument]) -> None:
        if not documents:
            raise ValueError("documents cannot be empty")
        module = importlib.import_module("rank_bm25")
        engine_type = module.BM25Okapi
        self.document_ids = [document.document_id for document in documents]
        self.engine: Any = engine_type(
            [tokenize(document.searchable_text) for document in documents]
        )

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores = np.asarray(self.engine.get_scores(tokenize(query)), dtype=float)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            RetrievalHit(
                document_id=self.document_ids[int(index)],
                score=float(scores[index]),
            )
            for index in order
        ]


class DenseBenchmarkIndex:
    def __init__(
        self,
        documents: Sequence[BenchmarkDocument],
        *,
        model_name: str,
        batch_size: int = 64,
    ) -> None:
        if not documents:
            raise ValueError("documents cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        sentence_transformers = importlib.import_module("sentence_transformers")
        faiss = importlib.import_module("faiss")
        model_type = sentence_transformers.SentenceTransformer
        self.model: Any = model_type(model_name)
        self.document_ids = [document.document_id for document in documents]
        encoded = self.model.encode(
            [document.searchable_text for document in documents],
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.embeddings = _normalize_rows(np.asarray(encoded, dtype=np.float32))
        self.document_positions = {
            document_id: index for index, document_id in enumerate(self.document_ids)
        }
        self.index: Any = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        encoded = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        normalized = _normalize_rows(np.asarray(encoded, dtype=np.float32))
        return np.asarray(normalized[0], dtype=np.float32)

    def search_embedding(
        self,
        query_embedding: NDArray[np.float32],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        count = min(top_k, len(self.document_ids))
        scores, indices = self.index.search(
            np.asarray([query_embedding], dtype=np.float32),
            count,
        )
        return [
            RetrievalHit(
                document_id=self.document_ids[int(index)],
                score=float(score),
            )
            for index, score in zip(indices[0], scores[0], strict=True)
            if index >= 0
        ]

    def embedding_for(self, document_id: str) -> NDArray[np.float32]:
        return np.asarray(
            self.embeddings[self.document_positions[document_id]],
            dtype=np.float32,
        )


def _normalize_rows(values: NDArray[np.float32]) -> NDArray[np.float32]:
    if values.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding matrix contains a zero vector")
    return np.asarray(
        np.ascontiguousarray(values / norms),
        dtype=np.float32,
    )
