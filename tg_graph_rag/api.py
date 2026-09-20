from functools import lru_cache

from llm.ag_llm import PipelineResult
from llm.rag_pipeline import run_rag, run_rag_async
from tg_graph_rag.config import (
    DEFAULT_METHOD,
    DEFAULT_TOP_K,
    DEFAULT_NUM_HOPS,
    DEFAULT_NUM_SEEN_MIN,
)


@lru_cache(maxsize=128)
def query(
    question: str,
    method: str = DEFAULT_METHOD,
    top_k: int = DEFAULT_TOP_K,
    num_hops: int = DEFAULT_NUM_HOPS,
    num_seen_min: int = DEFAULT_NUM_SEEN_MIN,
    model: str = "gpt-4.1-mini",
) -> PipelineResult:
    """Pipeline 3: semantic seed retrieval followed by graph traversal."""
    if method not in {"hybrid", "community"}:
        raise ValueError("method must be 'hybrid' or 'community'")
    return run_rag(
        question,
        mode="graph",
        top_k=top_k,
        model=model,
        retrieval_method=method,
        num_hops=num_hops,
        num_seen_min=num_seen_min,
    )


async def async_query(
    question: str,
    method: str = DEFAULT_METHOD,
    top_k: int = DEFAULT_TOP_K,
    num_hops: int = DEFAULT_NUM_HOPS,
    num_seen_min: int = DEFAULT_NUM_SEEN_MIN,
    model: str = "gpt-4.1-mini",
) -> PipelineResult:
    """Run GraphRAG with async embeddings and chat generation."""
    if method not in {"hybrid", "community"}:
        raise ValueError("method must be 'hybrid' or 'community'")
    return await run_rag_async(
        question,
        mode="graph",
        top_k=top_k,
        model=model,
        retrieval_method=method,
        num_hops=num_hops,
        num_seen_min=num_seen_min,
    )
