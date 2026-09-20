from functools import lru_cache

from llm.ag_llm import PipelineResult
from llm.rag_pipeline import run_rag, run_rag_async


@lru_cache(maxsize=128)
def query(
    question: str,
    top_k: int = 5,
    model: str = "gpt-4.1-mini",
) -> PipelineResult:
    """Pipeline 2: semantic vertex retrieval without graph traversal."""
    return run_rag(question, mode="basic", top_k=top_k, model=model)


async def async_query(
    question: str,
    top_k: int = 5,
    model: str = "gpt-4.1-mini",
) -> PipelineResult:
    """Run Basic RAG with async embeddings and chat generation."""
    return await run_rag_async(
        question,
        mode="basic",
        top_k=top_k,
        model=model,
    )


if __name__ == "__main__":
    question = "What diseases are associated with the gene BRCA1?"
    result = query(question)
    print("Answer:", result.answer)
    print("Prompt Tokens:", result.prompt_tokens)
    print("Completion Tokens:", result.completion_tokens)
    print("Total Tokens:", result.total_tokens)
    print("Latency (s):", result.latency_s)
    print("Cost ($):", result.cost)
