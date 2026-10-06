# Pre-defined GraphRAG Corpus

This workflow uses a dedicated `WikipediaGraph` graph and never writes to
`HetionetGraph`.

## Configure

Create `pre_defined/graphrag_deploy/.env` from `.env.example` and set:

- `TG_API_KEY` to the Savanna API token
- `LLM_API_KEY` to the custom LLM key
- `LLM_BASE_URL` to the custom endpoint

The endpoint should be the OpenAI-compatible API base URL. Set the matching
model names in `LLM_EMBEDDING_MODEL` and `LLM_COMPLETION_MODEL`.

Docker Compose passes these variables into the containers and renders
`configs/server_config.json` at container startup; the real values do not need
to be stored in the JSON file.

Set the same Savanna values for the CLI in the repository root `.env`:

```dotenv
TG_HOST=https://your-savanna-host
TG_API_KEY=your-tigergraph-token
TG_RESTPP_PORT=443
TG_GS_PORT=443
GRAPHRAG_HOST=http://localhost:8000
PREDEFINED_GRAPH=WikipediaGraph
```

Rotate any token that was previously committed to configuration.

## Start GraphRAG

```bash
cd pre_defined/graphrag_deploy
docker compose up -d
```

The UI is available at `http://localhost:3000` and the API at
`http://localhost:8000`.

## Initialize and load

From the repository root:

```bash
uv run python pre_defined/main.py --init
uv run python pre_defined/main.py --load --limit 10
uv run python pre_defined/main.py --load
uv run python pre_defined/main.py --refresh
```

`--load` without `--limit` submits all documents in `corpus.jsonl`.

## Query

```bash
uv run python pre_defined/main.py \
  --query "Which athletics event at the 2008 Summer Olympics had the most competitors?"
```

Use `--method similarity` or `--method community` to compare GraphRAG
retrieval modes.
## Benchmark pipelines

```bash
uv run python -m pre_defined.benchmark --dataset eval_public --limit 10
```

The benchmark compares five rows:

- `LLM-Only`: parametric baseline with no retrieval.
- `RAG`: Savanna similarity retrieval followed by LLM generation.
- `GraphRAG`: Savanna hybrid graph retrieval followed by LLM generation.
- `Agentic GraphRAG`: LLM planner/evaluator/synthesizer with similarity, graph,
  community and exhaustive `event_category_scan` retrieval tools.
- `Structured Graph`: exact graph-field count/rank baseline; no answer-generation LLM call.

RAG and GraphRAG do not receive a hidden structured-scan shortcut. Agentic GraphRAG
must select the scan tool for exhaustive count/rank and unique venue/date questions;
the benchmark trace records that choice and any coverage-policy override. Structured
Graph is applicable only to those supported question forms; other rows are marked N/A
and excluded from its accuracy denominator. This separates retrieval/generation quality
from the exact structured baseline.

LLM tokens and cost come from API responses. Graph retrieval time and estimated context
tokens are recorded separately. Per-operation traces, strategy changes, N/A status and
stop reasons are written to `pre_defined/data/benchmark_results.json`.
