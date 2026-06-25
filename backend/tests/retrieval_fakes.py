from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.retrieval.dense import FloatMatrix, FloatVector


class FakeEmbedder:
    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.vectors = vectors

    async def embed_query(self, query: str) -> tuple[float, ...]:
        return tuple(float(value) for value in self.vectors[query])

    async def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return np.asarray(
            [self.vectors[text] for text in texts],
            dtype=np.float32,
        )


class InMemoryVectorIndex:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: FloatMatrix) -> None:
        self.vectors = np.vstack([self.vectors, vectors])

    def search(
        self,
        query: FloatMatrix,
        limit: int,
    ) -> tuple[FloatMatrix, NDArray[np.int64]]:
        similarities = query @ self.vectors.T
        positions = np.argsort(-similarities, axis=1)[:, :limit]
        scores = np.take_along_axis(similarities, positions, axis=1)
        return (
            np.asarray(scores, dtype=np.float32),
            np.asarray(positions, dtype=np.int64),
        )

    def reconstruct(self, position: int) -> FloatVector:
        return np.asarray(self.vectors[position], dtype=np.float32)

    def save(self, path: Path) -> None:
        np.save(path, self.vectors)
