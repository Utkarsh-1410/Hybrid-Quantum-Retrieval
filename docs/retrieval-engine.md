# Retrieval Engine

The concrete retrieval implementation lives in:

- `backend/app/retrieval/bm25.py`
- `backend/app/retrieval/dense.py`
- `backend/app/retrieval/hybrid.py`

## Components

### BM25

`BM25Retriever` implements Okapi BM25 directly with configurable `k1` and `b`.
It indexes title and content, supports language/source/metadata filtering, and
persists its corpus and configuration to deterministic JSON.

### Dense retrieval

`SentenceTransformerEmbedder` lazily loads a Sentence Transformer model.
`DenseRetriever`:

1. encodes title and content
2. L2-normalizes vectors
3. inserts them into `faiss.IndexFlatIP`
4. uses normalized query vectors for exact cosine search
5. stores document metadata beside the FAISS index
6. reconstructs matched vectors for downstream quantum re-ranking

### Hybrid retrieval

`HybridRetriever` runs BM25 while the query embedding is generated, performs
dense search, independently min-max normalizes each score family, and computes:

```text
hybrid_score = 0.4 * normalized_bm25 + 0.6 * normalized_dense
```

The candidate multiplier defaults to five, allowing fusion to consider more
documents than the final requested result count.

## Dependencies

Core:

- Python 3.11-3.13
- NumPy
- Pydantic

Retrieval extra:

- `sentence-transformers`
- `faiss-cpu`

Install the backend and retrieval dependencies:

```bash
cd backend
python -m pip install -e ".[retrieval,dev]"
```

Use a CUDA-enabled PyTorch build before installing the retrieval extra when GPU
embedding inference is required. FAISS CPU remains suitable for exact search on
small and medium corpora.

## Execute

Run the complete example from the backend directory:

```bash
python scripts/retrieval_demo.py
```

The first execution downloads
`sentence-transformers/all-MiniLM-L6-v2`. The script builds both indexes, writes
them under `data/faiss`, and prints hybrid-ranked results.

Run unit tests:

```bash
pytest tests/test_bm25_retriever.py \
       tests/test_dense_retriever.py \
       tests/test_hybrid.py
```

The unit tests use a deterministic fake embedder and an in-memory exact vector
index, so they do not download models or require FAISS at test time.

## Programmatic usage

```python
embedder = SentenceTransformerEmbedder(
    "sentence-transformers/all-MiniLM-L6-v2"
)
bm25 = BM25Retriever(documents)
dense = DenseRetriever(embedder)
await dense.build(documents)

retriever = HybridRetriever(
    bm25=bm25,
    dense=dense,
    embedder=embedder,
)
results = await retriever.search(
    "quantum semantic search",
    limit=10,
    filters=SearchFilters(language="en"),
)
```

Persist and restore indexes:

```python
bm25.save(Path("data/faiss/bm25.json"))
dense.save(Path("data/faiss/documents.index"))

bm25 = BM25Retriever.load(Path("data/faiss/bm25.json"))
dense = DenseRetriever.load(
    Path("data/faiss/documents.index"),
    embedder=embedder,
)
```
