# SciFact Research Evaluation

## Objective

The benchmark tests whether normalized hybrid retrieval and Hilbert-space-
inspired contextual re-ranking improve retrieval quality over BM25 and dense
retrieval on the BEIR SciFact test collection.

The four evaluated systems are:

1. BM25
2. Sentence Transformer cosine retrieval with exact FAISS search
3. normalized hybrid fusion: `0.4 * BM25 + 0.6 * dense`
4. quantum-contextual re-ranking of the hybrid candidate set

## Reproducible run

```bash
cd backend
python -m pip install -e ".[benchmark]"
hybrid-search-benchmark \
  --cache-dir ../data/benchmarks \
  --output ../benchmark-results/scifact.json
```

For a short environment check:

```bash
hybrid-search-benchmark --max-queries 10
```

The command downloads the official BEIR SciFact archive when it is not already
cached. Dataset files, model artifacts, and generated reports are excluded from
Git.

## Method

- Corpus text is the concatenation of title and abstract.
- BM25 uses case-folded word tokenization.
- Dense vectors use `sentence-transformers/all-MiniLM-L6-v2`.
- Document vectors are L2-normalized before insertion into `IndexFlatIP`.
- BM25 and dense candidate scores are independently min-max normalized per
  query before fusion.
- Every method returns at most 100 documents.
- Quantum re-ranking receives exactly the hybrid top-100 candidate set.
- The contextual state is the normalized centroid of candidate embeddings.
- The final ranker combines hybrid prior, squared query-document state overlap,
  and candidate-context coherence.

## Metrics

The report includes:

- MAP
- MRR@10 and MRR@100
- Recall@10 and Recall@100
- nDCG@10 and nDCG@100
- paired per-query bootstrap deltas with 95% confidence intervals

The fixed candidate set isolates re-ranking effects. It does not measure whether
quantum scoring can recover relevant documents absent from hybrid top-100.

## Interpretation

`|<Q|D>|^2` is a quantum-inspired scoring analogy, not execution on quantum
hardware. Squaring also removes the sign of cosine similarity. Results should
therefore be treated as an empirical ranking hypothesis and compared against
strong controls, not as evidence of quantum advantage.

For publishable experiments, run at least three embedding models, tune weights
only on a validation collection, freeze the selected configuration, and report
test-set confidence intervals without further tuning.
