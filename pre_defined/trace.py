"""Per-operation trace shared by all three pre-defined pipelines.

LLM steps take their token usage, cost and latency straight from the API
response (`PipelineResult`). Retrieval steps against Savanna make no billed LLM
call that we can observe, so they only carry latency and a tiktoken estimate of
the context they returned.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

from llm.ag_llm import PipelineResult


@lru_cache(maxsize=16)
def _encoding(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    encoding = _encoding(model)
    if encoding is None:
        return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))
    return len(encoding.encode(text))


def truncate_tokens(text: str, max_tokens: int, model: str) -> str:
    encoding = _encoding(model)
    if encoding is None:
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return text if len(tokens) <= max_tokens else " ".join(tokens[:max_tokens]) + "\n[context truncated]"
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens]) + "\n[context truncated]"


@dataclass
class Step:
    index: int
    agent: str
    operation: str  # "llm" | "retrieval"
    tool: str = ""
    detail: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    context_tokens: int = 0
    chunks: int = 0
    new_chunks: int = 0


@dataclass
class Trace:
    steps: list[Step] = field(default_factory=list)
    strategy_changes: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

    def llm(self, agent: str, result: PipelineResult, detail: str = "") -> None:
        self.steps.append(Step(
            index=len(self.steps) + 1,
            agent=agent,
            operation="llm",
            tool=result.model,
            detail=detail,
            latency_s=result.latency_s,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost=result.cost,
        ))

    def retrieval(
        self, agent: str, tool: str, query: str, latency_s: float,
        context_tokens: int, chunks: int, new_chunks: int,
    ) -> None:
        self.steps.append(Step(
            index=len(self.steps) + 1,
            agent=agent,
            operation="retrieval",
            tool=tool,
            detail=query,
            latency_s=round(latency_s, 3),
            context_tokens=context_tokens,
            chunks=chunks,
            new_chunks=new_chunks,
        ))

    def change_strategy(self, kind: str, previous: str, current: str, why: str) -> None:
        self.strategy_changes.append({
            "step": len(self.steps), "type": kind, "from": previous, "to": current, "why": why,
        })

    @property
    def prompt_tokens(self) -> int:
        return sum(step.prompt_tokens for step in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(step.completion_tokens for step in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(step.total_tokens for step in self.steps)

    @property
    def cost(self) -> float:
        return round(sum(step.cost for step in self.steps), 6)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(step) for step in self.steps]
