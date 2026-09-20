from __future__ import annotations

import time
import os
import logging
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


PRICING = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


@dataclass
class PipelineResult:
    """Structured result for LLM queries, including token counts and cost."""
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    cost: float = 0.0
    model: str = ""
    retrieved_context: str = ""
    finish_reason: str = ""


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(exclude_none=True)
        except Exception:
            return {}
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            return {}
    return {}


def _read_field(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def extract_usage_metrics(payload: Any) -> dict[str, Any]:
    """Extract token and usage metadata from OpenAI or dict-like responses."""
    usage = _read_field(payload, "usage")
    usage_dict = _to_dict(usage)

    prompt_tokens = _read_field(usage, "prompt_tokens", usage_dict.get("prompt_tokens"))
    completion_tokens = _read_field(
        usage,
        "completion_tokens",
        usage_dict.get("completion_tokens"),
    )
    total_tokens = _read_field(usage, "total_tokens", usage_dict.get("total_tokens"))

    if prompt_tokens is None:
        prompt_tokens = _read_field(payload, "prompt_tokens", 0)
    if completion_tokens is None:
        completion_tokens = _read_field(payload, "completion_tokens", 0)
    if total_tokens is None:
        total_tokens = _read_field(payload, "total_tokens", 0)

    choices = _read_field(payload, "choices", []) or []
    first_choice = choices[0] if choices else None

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "model": str(_read_field(payload, "model", "") or ""),
        "finish_reason": str(_read_field(first_choice, "finish_reason", "") or ""),
    }


class LLMClient:
    """LLM client with token tracking and cost calculation."""
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_HOST_URL"),
            timeout=60,
        )
        self.async_client = AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_HOST_URL"),
            timeout=60,
        )
        self.model = model

    def query(self, messages: list[dict], temperature: float = 0) -> PipelineResult:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        latency = time.time() - start

        usage_data = extract_usage_metrics(response)
        prompt_tokens = usage_data["prompt_tokens"]
        completion_tokens = usage_data["completion_tokens"]
        total_tokens = usage_data["total_tokens"]

        pricing = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000

        choices = getattr(response, "choices", []) or []
        first_choice = choices[0] if choices else None
        message = getattr(first_choice, "message", None)
        answer = getattr(message, "content", "") or ""

        return PipelineResult(
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_s=round(latency, 3),
            cost=round(cost, 6),
            model=usage_data["model"] or self.model,
            finish_reason=usage_data["finish_reason"],
        )

    async def async_query(
        self,
        messages: list[dict],
        temperature: float = 0,
    ) -> PipelineResult:
        start = time.perf_counter()
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        latency = time.perf_counter() - start
        usage_data = extract_usage_metrics(response)
        prompt_tokens = usage_data["prompt_tokens"]
        completion_tokens = usage_data["completion_tokens"]
        total_tokens = usage_data["total_tokens"]
        pricing = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000
        message = response.choices[0].message if response.choices else None
        return PipelineResult(
            answer=getattr(message, "content", "") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_s=round(latency, 3),
            cost=round(cost, 6),
            model=usage_data["model"] or self.model,
            finish_reason=usage_data["finish_reason"],
        )

    def embed(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        response = self.client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
