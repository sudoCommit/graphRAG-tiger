from __future__ import annotations

import asyncio
import os
import time
from functools import lru_cache
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from llm.ag_llm import PRICING, PipelineResult
from tg.retrieval import SavannaRetriever
from tg_graph_rag.prompts import build_graph_rag_prompt
from tg_rag.prompts import build_rag_prompt


@lru_cache(maxsize=1)
def _shared_retriever() -> SavannaRetriever:
    return SavannaRetriever()


@lru_cache(maxsize=8)
def _chat_model(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_HOST_URL") or None,
        timeout=60,
    )


def _retrieve_context(
    retriever: SavannaRetriever,
    question: str,
    mode: Literal["basic", "graph"],
    top_k: int,
    num_hops: int,
    num_seen_min: int,
):
    result = retriever.retrieve(question, top_k)
    if mode == "graph":
        return retriever.expand(
            result.documents,
            num_hops=num_hops,
            num_seen_min=num_seen_min,
        )
    return result


def _build_messages(
    question: str,
    mode: Literal["basic", "graph"],
    context: str,
    retrieval_method: str,
) -> list[dict[str, str]]:
    if mode == "graph":
        return build_graph_rag_prompt(question, retrieval_method, context)
    return build_rag_prompt(question, context)


async def run_rag_async(
    question: str,
    *,
    mode: Literal["basic", "graph"],
    top_k: int,
    model: str,
    retrieval_method: str = "hybrid",
    num_hops: int = 0,
    num_seen_min: int = 1,
    retriever: SavannaRetriever | None = None,
    chat_model: BaseChatModel | None = None,
) -> PipelineResult:
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    start = time.perf_counter()
    active_retriever = retriever or _shared_retriever()
    retrieval_result = await active_retriever.aretrieve(question, top_k)
    if mode == "graph":
        retrieval_result = await active_retriever.aexpand(
            retrieval_result.documents,
            num_hops=num_hops,
            num_seen_min=num_seen_min,
        )
    messages = _build_messages(
        question,
        mode,
        retrieval_result.context,
        retrieval_method,
    )
    active_chat_model = chat_model or _chat_model(model)
    response = await active_chat_model.ainvoke(messages)
    return _result_from_response(
        response,
        model,
        retrieval_result.context,
        time.perf_counter() - start,
    )


def _result_from_response(
    response: Any,
    model: str,
    context: str,
    latency: float,
) -> PipelineResult:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage", {})
    prompt_tokens = int(
        usage.get("input_tokens") or token_usage.get("prompt_tokens") or 0
    )
    completion_tokens = int(
        usage.get("output_tokens") or token_usage.get("completion_tokens") or 0
    )
    total_tokens = int(
        usage.get("total_tokens")
        or token_usage.get("total_tokens")
        or prompt_tokens + completion_tokens
    )
    actual_model = str(
        metadata.get("model_name") or metadata.get("model") or model
    )
    pricing = PRICING.get(
        actual_model, PRICING.get(model, {"input": 0.0, "output": 0.0})
    )
    cost = (
        prompt_tokens * pricing["input"]
        + completion_tokens * pricing["output"]
    ) / 1_000_000
    return PipelineResult(
        answer=str(response.content),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_s=round(latency, 3),
        cost=round(cost, 6),
        model=actual_model,
        retrieved_context=context,
        finish_reason=str(metadata.get("finish_reason", "") or ""),
    )


def run_rag(
    question: str,
    *,
    mode: Literal["basic", "graph"],
    top_k: int,
    model: str,
    retrieval_method: str = "hybrid",
    num_hops: int = 0,
    num_seen_min: int = 1,
    retriever: SavannaRetriever | None = None,
    chat_model: BaseChatModel | None = None,
) -> PipelineResult:
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    start = time.perf_counter()
    active_retriever = retriever or _shared_retriever()
    retrieval_result = _retrieve_context(
        active_retriever,
        question,
        mode,
        top_k,
        num_hops,
        num_seen_min,
    )
    messages = _build_messages(
        question,
        mode,
        retrieval_result.context,
        retrieval_method,
    )

    active_chat_model = chat_model or _chat_model(model)
    response = active_chat_model.invoke(messages)
    latency = time.perf_counter() - start
    return _result_from_response(
        response,
        model,
        retrieval_result.context,
        latency,
    )
