from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkDocument:
    document_id: str
    title: str
    text: str

    @property
    def searchable_text(self) -> str:
        return f"{self.title} {self.text}".strip()


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    name: str
    corpus: dict[str, BenchmarkDocument]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    document_id: str
    score: float


@dataclass(frozen=True, slots=True)
class FusedHit:
    document_id: str
    bm25_score: float
    dense_score: float
    hybrid_score: float
