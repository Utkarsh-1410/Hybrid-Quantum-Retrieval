# Hybrid Quantum-Classical Retrieval System

A research-grade foundation for an AI-powered web search engine combining
lexical retrieval, dense vector search, Hilbert-space-inspired re-ranking, and
retrieval-augmented generation.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Docker Setup](#docker-setup)
  - [Local Setup](#local-setup)
- [arXiv Dataset Integration](#arxiv-dataset-integration)
- [API Documentation](#api-documentation)
- [Frontend Usage](#frontend-usage)
- [Development Workflow](#development-workflow)
- [Testing and Benchmarking](#testing-and-benchmarking)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)

## Overview

This repository contains the first implementation tranche:

- complete system architecture and engineering decisions
- production-oriented monorepo structure
- PostgreSQL and pgvector database schema
- FastAPI application skeleton with `/index`, `/search`, and `/ask`
- hybrid score fusion and quantum-inspired ranker
- concrete BM25, Sentence Transformer, and FAISS retrieval engine
- reproducible BEIR SciFact benchmark with statistical comparison
- unit tests for the ranking primitives
- Docker Compose development environment
- arXiv dataset integration for research-scale document retrieval

The external BM25, FAISS, Sentence Transformers, and LLM adapters are cleanly
separated from the API so they can be implemented and benchmarked independently.

## Architecture

The system separates durable content, retrieval indexes, ranking logic, and
answer generation. PostgreSQL is the system of record. BM25 and FAISS are
rebuildable search projections. This makes index recovery deterministic and
keeps model-specific artifacts out of transactional storage.

### System Components

- **PostgreSQL Database**: Stores documents, chunks, indexing jobs, search queries, and results with pgvector for vector similarity
- **FastAPI Backend**: RESTful API with endpoints for indexing, search, and RAG
- **React Frontend**: Modern web interface for search and chat interactions
- **BM25 Retriever**: Lexical search using rank-bm25
- **Dense Retriever**: Vector search using Sentence Transformers and FAISS
- **Quantum Ranker**: Hilbert-space-inspired re-ranking
- **RAG Pipeline**: LangChain-based retrieval-augmented generation

### Data Flow

1. **Indexing Flow**:
   - Validate and deduplicate documents using canonical URL and content hash
   - Persist the document and chunk records transactionally
   - Generate chunk embeddings in batches
   - Write lexical fields to the BM25 index
   - Append normalized vectors to the FAISS index
   - Commit index version metadata only after both projections succeed
   - Mark the indexing job complete; failed jobs remain retryable

2. **Search Flow**:
   - Normalize the query and detect optional filters
   - Run BM25 and dense retrieval concurrently with a larger candidate pool
   - Min-max normalize scores per retriever
   - Fuse candidates with `0.4 * BM25 + 0.6 * dense`
   - Build normalized query and document states
   - Compute quantum similarity `P = |<Q|D>|^2`
   - Blend quantum relevance with context coherence and the hybrid prior
   - Return ranked results with an explicit score explanation

3. **RAG Flow**:
   - `/ask` uses the same search pipeline
   - Applies evidence diversity and token budgeting
   - Sends a citation-aware prompt to the configured LLM
   - Returns generated answer with source documents, chunk identifiers, and confidence

## Project Structure

```
hybrid-quantum-search/
├── backend/
│   ├── app/
│   │   ├── api/              # HTTP endpoint handlers
│   │   │   ├── routes/       # Route implementations (index, search, ask, health)
│   │   │   ├── dependencies.py  # Dependency injection
│   │   │   └── router.py     # API router configuration
│   │   ├── core/             # Configuration, logging, errors
│   │   ├── db/               # SQLAlchemy models and sessions
│   │   ├── domain/           # Provider-neutral domain objects
│   │   ├── datasets/         # Streaming arXiv dataset adapter
│   │   ├── evaluation/       # SciFact benchmark and IR metrics
│   │   ├── quantum/          # State probability and contextual rankers
│   │   ├── rag/              # Retrieval, prompting, LLM, confidence
│   │   ├── ranking/          # Quantum-inspired re-ranking
│   │   ├── retrieval/        # BM25, Sentence Transformer, FAISS, hybrid
│   │   ├── schemas/          # Versioned API models
│   │   ├── services/         # Application orchestration
│   │   └── main.py           # FastAPI application factory
│   ├── scripts/              # Utility scripts
│   │   ├── build_arxiv_indexes.py  # Build arXiv search indexes
│   │   ├── quantum_ranking_demo.py
│   │   ├── rag_pipeline_demo.py
│   │   └── retrieval_demo.py
│   ├── tests/                # Unit and integration tests
│   ├── Dockerfile            # Backend container definition
│   └── pyproject.toml        # Python dependencies and project config
├── frontend/
│   ├── src/
│   │   ├── pages/            # Search, AI Chat, Analytics pages
│   │   ├── components/       # Reusable UI components
│   │   ├── lib/              # Utility libraries
│   │   ├── App.tsx           # Main React application
│   │   └── main.tsx          # Application entry point
│   ├── Dockerfile            # Frontend container definition
│   ├── package.json          # Node.js dependencies
│   └── vite.config.ts        # Vite build configuration
├── database/
│   └── schema.sql            # PostgreSQL database schema
├── docs/                     # Detailed documentation
│   ├── architecture.md       # System architecture details
│   ├── arxiv-dataset.md      # arXiv dataset integration
│   ├── quantum-ranking.md    # Quantum ranking mathematics
│   ├── rag-pipeline.md       # RAG implementation details
│   ├── repository-structure.md  # Repository organization
│   ├── research-evaluation.md  # Evaluation methodology
│   └── retrieval-engine.md   # Retrieval system details
├── .env.example              # Environment variables template
├── docker-compose.yml        # Docker orchestration
└── README.md                 # This file
```

## Setup Instructions

### Docker Setup (Recommended)

**Prerequisites**:
- Docker Desktop installed and running
- Git for cloning the repository

**Steps**:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hybrid-quantum-search
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration (optional for defaults)
   ```

3. Start the services:
   ```bash
   docker compose up --build
   ```

4. Access the services:
   - API documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`
   - Frontend: `http://localhost:3000`

5. Stop the services:
   ```bash
   docker compose down
   ```

### Local Setup

**Prerequisites**:
- Python 3.11+ (use `py` launcher on Windows)
- Node.js 18+ and npm
- PostgreSQL 16+ with pgvector extension
- Git for cloning the repository

**Steps**:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hybrid-quantum-search
   ```

2. Set up Python environment:
   ```bash
   # On Windows
   py -m pip install -e backend/[retrieval]

   # On Unix-like systems
   python -m pip install -e backend/[retrieval]
   ```

3. Set up PostgreSQL database:
   ```bash
   # Create database
   createdb hybrid_search

   # Install pgvector extension
   psql hybrid_search -c "CREATE EXTENSION vector;"

   # Run schema
   psql hybrid_search < database/schema.sql
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your database connection and other settings
   ```

5. Set up frontend:
   ```bash
   cd frontend
   npm install
   ```

6. Run backend:
   ```bash
   cd backend
   py -m uvicorn app.main:create_app --factory --reload --port 8000
   ```

7. Run frontend (in separate terminal):
   ```bash
   cd frontend
   npm run dev
   ```

8. Access the services:
   - API documentation: `http://localhost:8000/docs`
   - Frontend: `http://localhost:3000`

## arXiv Dataset Integration

The system includes integration with the arXiv research paper dataset for large-scale document retrieval experiments.

The dataset used for this project is Cornell University's arXiv metadata
snapshot from Kaggle: [Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).
It provides newline-delimited metadata records for arXiv papers, including
titles, abstracts, authors, categories, dates, DOI and journal metadata.

We used the dataset as the research corpus for retrieval experiments. The
loader streams the metadata file record by record, maps each paper title and
abstract into searchable documents, preserves arXiv categories and bibliographic
fields as metadata, and then builds BM25 and FAISS indexes for hybrid lexical
and dense retrieval. The generated indexes power both `/search` and `/ask`,
where retrieved arXiv papers are re-ranked with the quantum-inspired ranking
layer and used as evidence for RAG responses.

### Dataset Location

After downloading the Kaggle dataset, place the arXiv metadata snapshot at a
local path such as:
```
C:\Users\utkar\OneDrive\Desktop\build-a-complete-research-grade-project\work\arxiv-metadata-oai-snapshot.json
```

This is a ~5GB newline-delimited JSON file containing millions of research papers.

### Building arXiv Indexes

To build search indexes from the arXiv dataset:

1. **Install retrieval dependencies**:
   ```bash
   # With Docker: Already included in the Dockerfile
   # Local setup:
   py -m pip install -e backend/[retrieval]
   ```

2. **Build indexes**:
   ```bash
   cd backend

   # Build a bounded 10,000-paper index (default)
   py scripts/build_arxiv_indexes.py \
     "C:\Users\utkar\OneDrive\Desktop\build-a-complete-research-grade-project\work\arxiv-metadata-oai-snapshot.json"

   # Build an information-retrieval subset with 50,000 papers
   py scripts/build_arxiv_indexes.py \
     "C:\Users\utkar\OneDrive\Desktop\build-a-complete-research-grade-project\work\arxiv-metadata-oai-snapshot.json" \
     --category cs.IR \
     --max-documents 50000 \
     --output-dir ../data/faiss/arxiv-cs-ir
   ```

3. **Generated files**:
   - `bm25.json` - BM25 lexical index
   - `documents.index` - FAISS vector index
   - `documents.index.metadata.json` - Index metadata

4. **Configure API to use arXiv indexes**:
   ```bash
   # Update .env file
   BM25_INDEX_PATH=../data/faiss/arxiv/bm25.json
   FAISS_INDEX_PATH=../data/faiss/arxiv/documents.index
   ```

### arXiv Field Mapping

| arXiv field | Search field |
|---|---|
| `id` | external ID and canonical arXiv URL |
| `title` | document title |
| `abstract` | retrievable content |
| `authors` | metadata |
| `categories` | metadata and filtering |
| `update_date` | metadata and date filtering |
| DOI, journal reference, license, versions | metadata |

### Performance Considerations

- The full arXiv snapshot contains millions of papers
- Local exact FAISS and in-memory BM25 are appropriate for experiments and bounded subsets
- For production use, consider:
  - Distributed lexical indexing (OpenSearch)
  - Batched or sharded FAISS indexes
  - Checkpointed embedding jobs
  - Incremental metadata persistence

## API Documentation

### Endpoints

#### Health Check
- **GET** `/health`
- Returns system health status
- Response: `{"status": "ok"}`

#### Index Documents
- **POST** `/api/v1/index`
- Accepts one or more documents for asynchronous indexing
- Request body:
  ```json
  {
    "documents": [
      {
        "title": "Document title",
        "content": "Document content",
        "metadata": {}
      }
    ]
  }
  ```
- Returns indexing job ID for tracking

#### Search
- **POST** `/api/v1/search`
- Runs hybrid retrieval and quantum-inspired re-ranking
- Request body:
  ```json
  {
    "query": "search query",
    "top_k": 10,
    "filters": {}
  }
  ```
- Returns ranked results with scores and explanations

#### Ask (RAG)
- **POST** `/api/v1/ask`
- Retrieves evidence and returns a grounded answer
- Request body:
  ```json
  {
    "query": "question",
    "top_k": 10
  }
  ```
- Returns answer with sources, confidence score, and citations

### Interactive API Documentation

Once the system is running, access the interactive Swagger documentation:
- **URL**: `http://localhost:8000/docs`
- **Features**: Try out endpoints, view schemas, see examples

## Frontend Usage

### Pages

1. **Search Page** (`/`)
   - Hybrid search interface
   - Real-time result ranking
   - Score breakdown visualization

2. **Chat Page** (`/chat`)
   - AI-powered question answering
   - Citation-aware responses
   - Confidence indicators

3. **Analytics Page** (`/analytics`)
   - Search performance metrics
   - Retrieval statistics
   - System monitoring

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Type checking
npm run typecheck
```

## Development Workflow

### Backend Development

1. **Install development dependencies**:
   ```bash
   cd backend
   py -m pip install -e ".[dev]"
   ```

2. **Run tests**:
   ```bash
   pytest
   ```

3. **Code quality**:
   ```bash
   # Type checking
   mypy app/

   # Linting
   ruff check app/
   ```

4. **Run demo scripts**:
   ```bash
   # Retrieval demo
   py scripts/retrieval_demo.py

   # Quantum ranking demo
   py scripts/quantum_ranking_demo.py

   # RAG pipeline demo
   py scripts/rag_pipeline_demo.py
   ```

### Frontend Development

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Development server**:
   ```bash
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

### Testing and Benchmarking

#### Backend Tests

```bash
cd backend
pytest
```

#### SciFact Benchmark

```bash
cd backend
py -m pip install -e ".[benchmark]"
hybrid-search-benchmark \
  --cache-dir ../data/benchmarks \
  --output ../benchmark-results/scifact.json
```

This runs the system against the BEIR SciFact dataset and produces statistical comparison metrics.

## Troubleshooting

### Docker Issues

**Problem**: Docker Desktop not running
- **Solution**: Start Docker Desktop and wait for it to be fully initialized

**Problem**: Port conflicts (8000, 3000, 5432)
- **Solution**: Change ports in `docker-compose.yml` or stop conflicting services

**Problem**: Container build failures
- **Solution**: Check Docker logs, ensure sufficient disk space, try `docker compose build --no-cache`

### Python Issues

**Problem**: Module not found errors
- **Solution**: Ensure dependencies are installed with `py -m pip install -e backend/[retrieval]`

**Problem**: TensorFlow/Keras compatibility
- **Solution**: Install `tf-keras` package: `py -m pip install tf-keras`

**Problem**: FAISS installation errors
- **Solution**: Use CPU-only version: `py -m pip install faiss-cpu`

### Database Issues

**Problem**: PostgreSQL connection refused
- **Solution**: Ensure PostgreSQL is running and accessible, check connection string in `.env`

**Problem**: pgvector extension not found
- **Solution**: Install pgvector extension in PostgreSQL: `CREATE EXTENSION vector;`

### arXiv Dataset Issues

**Problem**: Memory errors during index building
- **Solution**: Reduce `--max-documents` parameter, increase system RAM, or process in batches

**Problem**: Slow embedding generation
- **Solution**: Use GPU if available, reduce batch size, or use a smaller embedding model

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Application
APP_ENV=development
LOG_LEVEL=INFO
API_V1_PREFIX=/api/v1

# Database
DATABASE_URL=postgresql+asyncpg://search:search@postgres:5432/hybrid_search
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Index Paths
BM25_INDEX_PATH=/data/faiss/bm25.json
FAISS_INDEX_PATH=/data/faiss/documents.index

# Search Parameters
DEFAULT_TOP_K=10
MAX_TOP_K=100
BM25_WEIGHT=0.4
DENSE_WEIGHT=0.6
RANKER_HYBRID_WEIGHT=0.35
QUANTUM_WEIGHT=0.45
CONTEXT_WEIGHT=0.20

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_TEMPERATURE=0
OPENAI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434

# RAG Parameters
RAG_MAX_CHARS_PER_SOURCE=4000
RAG_MAX_CONTEXT_CHARS=16000
RAG_MINIMUM_EVIDENCE=3

# Indexing
INDEXING_BATCH_SIZE=64
```

### Customizing Ranking Weights

The hybrid search uses weighted score fusion. Adjust weights in `.env`:

- `BM25_WEIGHT`: Lexical search importance (default: 0.4)
- `DENSE_WEIGHT`: Vector search importance (default: 0.6)
- `QUANTUM_WEIGHT`: Quantum ranking importance (default: 0.45)
- `CONTEXT_WEIGHT`: Context coherence importance (default: 0.20)

## Additional Documentation

For more detailed information on specific components:

- **System Architecture**: [docs/architecture.md](docs/architecture.md)
- **Repository Structure**: [docs/repository-structure.md](docs/repository-structure.md)
- **Research Evaluation**: [docs/research-evaluation.md](docs/research-evaluation.md)
- **Retrieval Engine**: [docs/retrieval-engine.md](docs/retrieval-engine.md)
- **Quantum Ranking**: [docs/quantum-ranking.md](docs/quantum-ranking.md)
- **RAG Pipeline**: [docs/rag-pipeline.md](docs/rag-pipeline.md)
- **arXiv Dataset**: [docs/arxiv-dataset.md](docs/arxiv-dataset.md)

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines here]

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation in the `docs/` directory
- Review troubleshooting section above
