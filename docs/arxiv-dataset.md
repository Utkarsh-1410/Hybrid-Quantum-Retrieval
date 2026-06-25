# arXiv Dataset Integration

The supplied dataset is:

```text
C:\Users\utkar\Documents\Codex\2026-06-13\
build-a-complete-research-grade-project\work\
arxiv-metadata-oai-snapshot.json
```

It is a roughly 5 GiB newline-delimited JSON snapshot. Each line contains one
paper. The loader therefore streams one line at a time and never reads the whole
file into memory.

## Mapping

| arXiv field | Search field |
|---|---|
| `id` | external ID and canonical arXiv URL |
| `title` | document title |
| `abstract` | retrievable content |
| `authors` | metadata |
| `categories` | metadata and filtering |
| `update_date` | metadata and date filtering |
| DOI, journal reference, license, versions | metadata |

Document and chunk UUIDs are deterministically derived from the arXiv ID, so
repeated ingestion produces the same identifiers.

## Build Local Indexes

Install retrieval dependencies:

```powershell
cd backend
python -m pip install -e ".[retrieval]"
```

Build a bounded 10,000-paper index:

```powershell
python scripts/build_arxiv_indexes.py `
  "C:\Users\utkar\Documents\Codex\2026-06-13\build-a-complete-research-grade-project\work\arxiv-metadata-oai-snapshot.json"
```

Build an information-retrieval subset:

```powershell
python scripts/build_arxiv_indexes.py `
  "C:\Users\utkar\Documents\Codex\2026-06-13\build-a-complete-research-grade-project\work\arxiv-metadata-oai-snapshot.json" `
  --category cs.IR `
  --max-documents 50000 `
  --output-dir ../data/faiss/arxiv-cs-ir
```

The command writes:

- `bm25.json`
- `documents.index`
- `documents.index.metadata.json`

Point the API at those files:

```powershell
$env:BM25_INDEX_PATH="../data/faiss/arxiv-cs-ir/bm25.json"
$env:FAISS_INDEX_PATH="../data/faiss/arxiv-cs-ir/documents.index"
uvicorn app.main:create_app --factory --reload
```

When these files are present, both `/search` and `/ask` automatically use the
persisted arXiv indexes.

## Scale Note

The snapshot contains millions of papers. The local exact FAISS and in-memory
BM25 implementation is appropriate for experiments and bounded subsets. A full
production index should use:

- distributed lexical indexing such as OpenSearch
- batched or sharded FAISS indexes
- checkpointed embedding jobs
- incremental metadata persistence

The CLI defaults to 10,000 papers to prevent an accidental multi-hour,
high-memory build. Increase the limit deliberately after measuring available
RAM, disk, and embedding throughput.
