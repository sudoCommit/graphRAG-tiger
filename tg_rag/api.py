import time
import logging

import tiktoken

from llm.ag_llm import PipelineResult, PRICING, extract_usage_metrics
from tg.base import get_graphrag_connection

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
    top_k: int = 5,
    model: str = "gpt-4.1-mini",
) -> PipelineResult:
    """Pipeline 2: Basic RAG — vector-only retrieval (no graph traversal).

    Uses the same embeddings stored by the GraphRAG service in TigerGraph,
    but sets num_hops=0 so only vector similarity search is performed.
    No multi-hop graph traversal.
    """
    conn = get_graphrag_connection()

    start = time.time()
    resp = conn.ai.answerQuestion(
        question,
        method="hybrid",
        method_parameters={
            "top_k": top_k,
            "num_hops": 0,
            "verbose": True,
        },
    )
    latency = time.time() - start

    answer = resp.get("response", str(resp))
    context = resp.get("retrieved_context", "")
    usage_data = extract_usage_metrics(resp)

    # Try to extract token counts from response; estimate if absent
    prompt_tokens = usage_data["prompt_tokens"]
    completion_tokens = usage_data["completion_tokens"]
    if not prompt_tokens:
        prompt_tokens = _estimate_tokens(str(context) + question)
        completion_tokens = _estimate_tokens(answer)

    total_tokens = usage_data["total_tokens"] or (prompt_tokens + completion_tokens)

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
        model=usage_data["model"] or model,
        retrieved_context=str(context),
        response_id=usage_data["response_id"],
        created=usage_data["created"],
        service_tier=usage_data["service_tier"],
        system_fingerprint=usage_data["system_fingerprint"],
        finish_reason=usage_data["finish_reason"],
        completion_tokens_details=usage_data["completion_tokens_details"],
        prompt_tokens_details=usage_data["prompt_tokens_details"],
        latency_checkpoint=usage_data["latency_checkpoint"],
    )
