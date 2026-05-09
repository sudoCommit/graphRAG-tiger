
import time
import os
import logging
from dataclasses import dataclass

from openai import OpenAI
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


class LLMClient:
    """LLM client with token tracking and cost calculation."""
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(
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

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        pricing = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000

        return PipelineResult(
            answer=response.choices[0].message.content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_s=round(latency, 3),
            cost=round(cost, 6),
            model=self.model,
        )

    def embed(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        response = self.client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
