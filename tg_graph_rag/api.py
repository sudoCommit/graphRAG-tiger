import time
import logging

import tiktoken

from llm.ag_llm import PipelineResult, PRICING
from tg.base import get_graphrag_connection
from tg_graph_rag.config import (
    DEFAULT_METHOD,
    DEFAULT_TOP_K,
    DEFAULT_NUM_HOPS,
    DEFAULT_NUM_SEEN_MIN,
)

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def query(
    question: str,
    method: str = DEFAULT_METHOD,
    top_k: int = DEFAULT_TOP_K,
    num_hops: int = DEFAULT_NUM_HOPS,
    num_seen_min: int = DEFAULT_NUM_SEEN_MIN,
    model: str = "gpt-4.1-mini",
) -> PipelineResult:
    """Pipeline 3: GraphRAG — hybrid retrieval (vector + graph traversal).

    Uses the deployed TigerGraph GraphRAG service with multi-hop graph
    traversal on top of vector similarity search.
    """
    conn = get_graphrag_connection()

    method_parameters = {
        "top_k": top_k,
        "num_hops": num_hops,
        "num_seen_min": num_seen_min,
        "verbose": True,
    }

    if method == "hybrid":
        method_parameters["indices"] = ["DocumentChunk", "Community"]
    elif method == "community":
        method_parameters["community_level"] = 2
        method_parameters["combine"] = False

    start = time.time()
    resp = conn.ai.answerQuestion(
        question,
        method=method,
        method_parameters=method_parameters,
    )
    latency = time.time() - start

    answer = resp.get("response", str(resp))
    context = resp.get("retrieved_context", "")

    # Try to extract token counts from response; estimate if absent
    prompt_tokens = resp.get("prompt_tokens", 0)
    completion_tokens = resp.get("completion_tokens", 0)
    if not prompt_tokens:
        prompt_tokens = _estimate_tokens(str(context) + question)
        completion_tokens = _estimate_tokens(answer)

    total_tokens = prompt_tokens + completion_tokens

    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (
        prompt_tokens * pricing["input"]
        + completion_tokens * pricing["output"]
    ) / 1_000_000

    return PipelineResult(
        answer=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_s=round(latency, 3),
        cost=round(cost, 6),
        model=model,
        retrieved_context=str(context),
    )
