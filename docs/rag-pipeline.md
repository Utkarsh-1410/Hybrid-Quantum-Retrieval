# RAG Pipeline

## Flow

```text
Query
  -> normalized BM25 + dense hybrid retrieval
  -> quantum-contextual re-ranking
  -> diversified top documents
  -> citation-aware grounded prompt
  -> LangChain chat model
  -> answer + validated citations + confidence
```

## Modules

- `backend/app/rag/retriever.py`
  - requests a larger hybrid candidate pool
  - applies quantum-contextual ranking
  - limits repeated chunks from the same document
  - assigns stable prompt citation IDs such as `[S1]`
- `backend/app/rag/generator.py`
  - constructs a bounded grounded prompt
  - supports OpenAI through `langchain-openai`
  - supports local Llama models through Ollama and `langchain-ollama`
  - validates model-emitted citation IDs against retrieved evidence
- `backend/app/rag/pipeline.py`
  - orchestrates retrieval and generation
  - returns answer, citations, all evidence, and confidence

## Installation

```bash
cd backend
python -m pip install -e ".[retrieval,rag,dev]"
```

## OpenAI

```bash
set LLM_PROVIDER=openai
set LLM_MODEL=gpt-4.1-mini
set OPENAI_API_KEY=your-key
python scripts/rag_pipeline_demo.py
```

On PowerShell use `$env:NAME="value"` for environment variables.

## Local Llama with Ollama

Install Ollama, fetch a Llama model, and start its service:

```bash
ollama pull llama3.1
ollama serve
```

Then run:

```bash
set LLM_PROVIDER=ollama
set LLM_MODEL=llama3.1
python scripts/rag_pipeline_demo.py
```

## Grounding and citations

Evidence blocks are labeled `[S1]`, `[S2]`, and so on. The prompt requires the
model to cite factual claims with those exact IDs. IDs not present in retrieved
evidence are removed from the structured citation output.

The answer text is preserved verbatim, including any invalid citation text, so
auditing can detect model noncompliance. Applications should render only the
validated structured citations as trusted links.

## Confidence

Confidence is a bounded heuristic:

```text
0.65 * mean top evidence score
+ 0.15 * evidence coverage
+ 0.20 * citation support
```

It is not a calibrated probability of factual correctness. Production systems
should calibrate it against a labeled answer-quality dataset.

## Tests

```bash
pytest tests/test_rag_retriever.py \
       tests/test_rag_generator.py \
       tests/test_rag_pipeline.py
```

Tests use deterministic fake chat and retrieval models. They do not call OpenAI
or require an Ollama server.
