from llm.ag_llm import PipelineResult
from llm.rag_pipeline import run_rag


async def async_query(
    question: str,
    top_k: int = 5,
    model: str = "gpt-4.1-mini",
) -> PipelineResult:
    """Run Basic RAG with async embeddings and chat generation."""
    return await run_rag(
        question,
        mode="basic",
        top_k=top_k,
        model=model,
    )
