import asyncio
import logging
import os
import sys

from llm.ag_llm import PipelineResult
import tg_llm.api as LlmPipeline
import tg_rag.api as RagPipeline
import tg_graph_rag.api as GraphRagPipeline


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_comparison(question: str, model: str = "gpt-4.1-mini") -> dict[str, PipelineResult]:
    """Run the same question through all 3 pipelines and print comparison."""
    logger.info("\n%s\nQuestion: %s\n%s", "=" * 80, question, "=" * 80)

    outcomes = asyncio.run(_run_pipelines(question, model))
    results: dict[str, PipelineResult] = {}
    labels = ("LLM-Only", "Basic RAG", "GraphRAG")
    for label, outcome in zip(labels, outcomes):
        logger.info("Running %s...", label)
        if isinstance(outcome, Exception):
            logger.error("%s failed: %s", label, outcome)
        else:
            results[label] = outcome
            logger.info("%s done (%ss)", label, outcome.latency_s)

    # Comparison table
    _print_comparison(results)
    return results


async def _run_pipelines(
    question: str,
    model: str,
) -> tuple[
    PipelineResult | Exception,
    PipelineResult | Exception,
    PipelineResult | Exception,
]:
    outcomes = await asyncio.gather(
        LlmPipeline.async_query(question, model=model),
        RagPipeline.async_query(question, model=model),
        GraphRagPipeline.async_query(question, model=model),
        return_exceptions=True,
    )
    return outcomes[0], outcomes[1], outcomes[2]


def _print_comparison(results: dict[str, PipelineResult]):
    names = ["LLM-Only", "Basic RAG", "GraphRAG"]

    lines = [f"\n{'─'*80}", f"{'Metric':<20}" + "".join(f"{name:>18}" for name in names), f"{'─'*80}"]

    for label, key in [
        ("Prompt Tokens", "prompt_tokens"),
        ("Completion Tokens", "completion_tokens"),
        ("Total Tokens", "total_tokens"),
        ("Latency (s)", "latency_s"),
        ("Cost ($)", "cost"),
    ]:
        line = f"{label:<20}"
        for name in names:
            if name in results:
                v = getattr(results[name], key)
                if isinstance(v, float):
                    line += f"{v:>18.4f}"
                else:
                    line += f"{v:>18}"
            else:
                line += f"{'N/A':>18}"
        lines.append(line)

    lines.append(f"{'─'*80}")
    logger.info("\n".join(lines))

    # Answers
    for name in names:
        if name in results:
            logger.info("\n%s answer:\n%s", name, results[name].answer[:500])

            detail_lines: list[str] = []
            if results[name].finish_reason:
                detail_lines.append(f"finish_reason={results[name].finish_reason}")

            if detail_lines:
                logger.info("%s usage details: %s", name, ", ".join(detail_lines))

    # Token reduction metric
    if "Basic RAG" in results and "GraphRAG" in results:
        rag = results["Basic RAG"].total_tokens
        graph = results["GraphRAG"].total_tokens
        if rag > 0:
            reduction = ((rag - graph) / rag) * 100
            logger.info("Token reduction (GraphRAG vs Basic RAG): %.1f%%", reduction)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else (
        "What compounds treat epilepsy and what are their side effects?"
    )
    run_comparison(q)
