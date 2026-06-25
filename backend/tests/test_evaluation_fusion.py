import pytest

from app.evaluation.fusion import fuse_hits
from app.evaluation.models import RetrievalHit


def test_fuse_hits_normalizes_retriever_scales() -> None:
    lexical = [
        RetrievalHit("lexical", 20.0),
        RetrievalHit("shared", 10.0),
    ]
    dense = [
        RetrievalHit("dense", 0.9),
        RetrievalHit("shared", 0.1),
    ]

    results = fuse_hits(lexical, dense)
    scores = {result.document_id: result.hybrid_score for result in results}

    assert scores["lexical"] == pytest.approx(0.4)
    assert scores["dense"] == pytest.approx(0.6)
    assert scores["shared"] == pytest.approx(0.0)
