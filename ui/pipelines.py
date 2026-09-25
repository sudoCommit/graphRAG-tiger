import asyncio

from llm.ag_llm import PipelineResult
import custom.tg_llm.api as LlmPipeline
import custom.tg_rag.api as RagPipeline
import custom.tg_graph_rag.api as GraphRagPipeline


async def run_all(
    question: str,
    model: str,
    top_k: int,
    num_hops: int,
    num_seen_min: int,
    edge_limit: int,
    slots: dict[str, object],
    render_result,
) -> tuple[dict[str, PipelineResult], dict[str, Exception]]:
    async def run_one(name: str, task):
        try:
            return name, await task
        except Exception as error:
            return name, error

    tasks = [
        asyncio.create_task(run_one("LLM-Only", LlmPipeline.async_query(question, model=model))),
        asyncio.create_task(run_one("Basic RAG", RagPipeline.async_query(question, top_k=top_k, model=model))),
        asyncio.create_task(run_one("GraphRAG", GraphRagPipeline.async_query(
            question, top_k=top_k, num_hops=num_hops, num_seen_min=num_seen_min,
            edge_limit=edge_limit, model=model
        ))),
    ]
    results: dict[str, PipelineResult] = {}
    errors: dict[str, Exception] = {}
    for completed in asyncio.as_completed(tasks):
        name, outcome = await completed
        if isinstance(outcome, Exception):
            errors[name] = outcome
            render_result(slots[name], name, "error", error=outcome)
        else:
            results[name] = outcome
            render_result(slots[name], name, "done", result=outcome)
    return results, errors
