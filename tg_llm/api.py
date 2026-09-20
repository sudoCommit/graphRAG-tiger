from functools import lru_cache

from llm.ag_llm import LLMClient, PipelineResult
from tg_llm.prompts import build_llm_only_prompt


@lru_cache(maxsize=128)
def query(question: str, model: str = "gpt-4.1-mini") -> PipelineResult:
    """Pipeline 1: LLM-Only. No retrieval, pure parametric knowledge."""
    client = _client(model)
    messages = build_llm_only_prompt(question)
    return client.query(messages)


@lru_cache(maxsize=8)
def _client(model: str) -> LLMClient:
    return LLMClient(model=model)


async def async_query(
    question: str,
    model: str = "gpt-4.1-mini",
) -> PipelineResult:
    """Run the LLM pipeline through the native asynchronous OpenAI client."""
    client = _client(model)
    return await client.async_query(build_llm_only_prompt(question))
