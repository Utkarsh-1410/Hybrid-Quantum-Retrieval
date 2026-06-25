from fastapi.testclient import TestClient

from app.api.dependencies import get_indexer
from app.main import create_app
from app.services.stubs import PendingDocumentIndexer


def test_health_endpoint() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_accepts_documents() -> None:
    app = create_app()
    app.dependency_overrides[get_indexer] = PendingDocumentIndexer
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/index",
            json={
                "documents": [
                    {
                        "title": "Quantum retrieval",
                        "content": "A test document.",
                    }
                ]
            },
        )
    assert response.status_code == 202
    assert response.json()["accepted_count"] == 1


def test_search_returns_empty_result_for_empty_corpus() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/search",
            json={"query": "quantum-inspired ranking", "top_k": 5},
        )
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_ask_returns_grounded_empty_corpus_message() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/ask",
            json={"query": "How does the ranker work?", "top_k": 3},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == 0.0
    assert body["sources"] == []
    assert "No indexed evidence" in body["answer"]
