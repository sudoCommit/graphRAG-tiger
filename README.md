# graphRAG-tiger

Custom LangChain/LangGraph RAG pipelines over an existing TigerGraph Savanna
graph. The RAG application reads an existing graph; the `tg/tg_load.py` utility
can explicitly create, drop, reset, and load the Hetionet schema from the CLI.

## Configuration

```dotenv
TG_HOST=https://your-savanna-host
TG_GRAPH=YourExistingGraph
TG_API_KEY=your-token
LLM_API_KEY=your-openai-compatible-key
LLM_HOST_URL=https://api.openai.com/v1
```

Optional retrieval settings:

- `TG_VERTEX_TYPES`: comma-separated vertex types to search; defaults to all.
- `TG_CANDIDATES_PER_TYPE`: vertices read per type and cached; defaults to 200.
- `TG_EDGES_PER_VERTEX`: outgoing edges read during expansion; defaults to 25.
- `EMBEDDING_MODEL`: embedding model; defaults to `text-embedding-3-small`.

## Run

```bash
uv sync --extra dev
uv run python -m tg.tiger_graph
uv run streamlit run app.py
```

`tg.tiger_graph` only validates the configured graph and reports vertex counts.
Basic RAG performs semantic vertex retrieval. Graph RAG reuses those seeds and
adds bounded multi-hop neighborhood traversal.

## Graph Management

```bash
uv run python tg/tg_load.py --schema
uv run python tg/tg_load.py --limit 2
uv run python tg/tg_load.py --load-data --limit 2
uv run python tg/tg_load.py --limit 2 --reset
uv run python tg/tg_load.py --drop-graph
uv run python tg/tg_load.py --reset-savanna
```

`--limit` loads a small sample per vertex and edge type into an existing graph.
Combine `--schema` or `--reset` with `--limit` to create or recreate the schema
before loading data. `--reset` drops and recreates the configured `TG_GRAPH`.


## Architecture
```mermaid
sequenceDiagram
    participant UI as app.py (Streamlit)
    participant P1 as tg_llm.api (LLM-Only)
    participant P2 as tg_rag.api (Basic RAG)
    participant P3 as tg_graph_rag.api (GraphRAG)
    participant RAG as llm/rag_pipeline.py
    participant RET as tg/retrieval.py (SavannaRetriever)
    participant TG as TigerGraph
    participant LLM as OpenAI/LangChain

    UI->>UI: asyncio.run(_run_all_pipelines)
    par concurrent tasks
        UI->>P1: pipeline1.async_query(question, model)
        P1->>LLM: LLMClient.async_query(messages)
        LLM-->>P1: PipelineResult
    and
        UI->>P2: pipeline2.async_query(question, top_k, model)
        P2->>RAG: run_rag_async(mode="basic", ...)
        RAG->>RET: aretrieve(question, top_k)
        RET->>TG: embed + vector search + fetch vertices
        TG-->>RET: documents
        RAG->>LLM: chat_model.ainvoke(messages)
        LLM-->>P2: PipelineResult
    and
        UI->>P3: pipeline3.async_query(question, top_k, num_hops, num_seen_min, model)
        P3->>RAG: run_rag_async(mode="graph", ...)
        RAG->>RET: aretrieve then aexpand (graph hops)
        RET->>TG: vector search + getEdges traversal
        TG-->>RET: seed + expanded documents
        RAG->>LLM: chat_model.ainvoke(messages)
        LLM-->>P3: PipelineResult
    end
    UI->>UI: asyncio.as_completed renders each card as it finishes
```
