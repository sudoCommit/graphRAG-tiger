from llm.ag_llm import LLMClient, PipelineResult
from tg_llm.prompts import build_llm_only_prompt


def query(question: str, model: str = "gpt-4.1-mini") -> PipelineResult:
    """Pipeline 1: LLM-Only. No retrieval, pure parametric knowledge."""
    client = LLMClient(model=model)
    messages = build_llm_only_prompt(question)
    return client.query(messages)
