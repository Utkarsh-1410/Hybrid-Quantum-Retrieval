# Repository Structure

```text
hybrid-quantum-search/
|-- backend/
|   |-- app/
|   |   |-- api/
|   |   |   |-- routes/          # HTTP endpoint handlers
|   |   |   |-- dependencies.py  # dependency injection
|   |   |   `-- router.py
|   |   |-- core/                # configuration, logging, errors
|   |   |-- db/                  # SQLAlchemy models and sessions
|   |   |-- domain/              # provider-neutral domain objects
|   |   |-- datasets/            # streaming arXiv dataset adapter
|   |   |-- evaluation/          # SciFact benchmark and IR metrics
|   |   |-- quantum/             # state probability and contextual rankers
|   |   |-- rag/                 # retrieval, prompting, LLM, confidence
|   |   |-- ranking/             # quantum-inspired re-ranking
|   |   |-- retrieval/           # BM25, Sentence Transformer, FAISS, hybrid
|   |   |-- schemas/             # versioned API models
|   |   |-- services/            # application orchestration
|   |   `-- main.py
|   |-- tests/
|   |-- Dockerfile
|   `-- pyproject.toml
|-- database/
|   `-- schema.sql
|-- docs/
|   |-- architecture.md
|   `-- repository-structure.md
|-- frontend/
|   |-- src/
|   |   |-- pages/               # Search, AI Chat, Analytics
|   |   |-- components/
|   |   `-- lib/
|   |-- Dockerfile
|   `-- package.json
|-- .env.example
|-- docker-compose.yml
`-- README.md
```

The domain and ranking packages deliberately have no FastAPI or database
dependency. They can be benchmarked in notebooks, batch jobs, and offline
evaluation pipelines without booting the web application.
