from llm.ag_llm import PipelineResult
from llm.rag_pipeline import run_rag
from custom.tg_graph_rag.config import (
    DEFAULT_TOP_K,
    DEFAULT_NUM_HOPS,
    DEFAULT_NUM_SEEN_MIN,
    DEFAULT_EDGE_LIMIT,
)


async def async_query(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    num_hops: int = DEFAULT_NUM_HOPS,
    num_seen_min: int = DEFAULT_NUM_SEEN_MIN,
    edge_limit: int = DEFAULT_EDGE_LIMIT,
    model: str = "gpt-4.1-mini",
) -> PipelineResult:
    """Run GraphRAG with async embeddings and chat generation."""
    return await run_rag(
        question,
        mode="graph",
        top_k=top_k,
        model=model,
        num_hops=num_hops,
        num_seen_min=num_seen_min,
        edge_limit=edge_limit,
    )
