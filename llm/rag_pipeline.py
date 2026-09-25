import time
from functools import lru_cache
from typing import Literal

from llm.ag_llm import LLMClient, PipelineResult
from custom.tg.retrieval import SavannaRetriever
from custom.tg_graph_rag.prompts import build_graph_rag_prompt
from custom.tg_rag.prompts import build_rag_prompt


@lru_cache(maxsize=1)
def _shared_retriever() -> SavannaRetriever:
    return SavannaRetriever()


@lru_cache(maxsize=8)
def _llm_client(model: str) -> LLMClient:
    return LLMClient(model=model)


def _build_messages(
    question: str,
    mode: Literal["basic", "graph"],
    context: str,
) -> list[dict[str, str]]:
    if mode == "graph":
        return build_graph_rag_prompt(question, context)
    return build_rag_prompt(question, context)


async def run_rag(
    question: str,
    *,
    mode: Literal["basic", "graph"],
    top_k: int,
    model: str,
    num_hops: int = 0,
    num_seen_min: int = 1,
    edge_limit: int | None = None,
    retriever: SavannaRetriever | None = None,
    llm_client: LLMClient | None = None,
) -> PipelineResult:
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    start = time.perf_counter()
    active_retriever = retriever or _shared_retriever()
    retrieval_result = await active_retriever.retrieve(question, top_k)
    if mode == "graph":
        retrieval_result = await active_retriever.expand(
            retrieval_result.documents,
            num_hops=num_hops,
            num_seen_min=num_seen_min,
            edge_limit=edge_limit,
        )

    messages = _build_messages(
        question,
        mode,
        retrieval_result.context,
    )
    active_llm_client = llm_client or _llm_client(model)
    result = await active_llm_client.query(messages)
    result.retrieved_context = retrieval_result.context
    result.latency_s = round(time.perf_counter() - start, 3)

    return result
