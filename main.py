import asyncio
import sys

from llm.ag_llm import PipelineResult
import tg_llm.api as pipeline1
import tg_rag.api as pipeline2
import tg_graph_rag.api as pipeline3


def run_comparison(question: str, model: str = "gpt-4.1-mini") -> dict[str, PipelineResult]:
    """Run the same question through all 3 pipelines and print comparison."""
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"{'='*80}\n")

    outcomes = asyncio.run(_run_pipelines(question, model))
    results: dict[str, PipelineResult] = {}
    labels = ("LLM-Only", "Basic RAG", "GraphRAG")
    for label, outcome in zip(labels, outcomes):
        print(f"▶ {label}...")
        if isinstance(outcome, Exception):
            print(f"  ❌ Failed: {outcome}")
        else:
            results[label] = outcome
            print(f"  ✅ Done ({outcome.latency_s}s)")

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
        pipeline1.async_query(question, model=model),
        pipeline2.async_query(question, model=model),
        pipeline3.async_query(question, model=model),
        return_exceptions=True,
    )
    return outcomes[0], outcomes[1], outcomes[2]


def _print_comparison(results: dict[str, PipelineResult]):
    names = ["LLM-Only", "Basic RAG", "GraphRAG"]

    print(f"\n{'─'*80}")
    print(f"{'Metric':<20}", end="")
    for name in names:
        print(f"{name:>18}", end="")
    print(f"\n{'─'*80}")

    for label, key in [
        ("Prompt Tokens", "prompt_tokens"),
        ("Completion Tokens", "completion_tokens"),
        ("Total Tokens", "total_tokens"),
        ("Latency (s)", "latency_s"),
        ("Cost ($)", "cost"),
    ]:
        print(f"{label:<20}", end="")
        for name in names:
            if name in results:
                v = getattr(results[name], key)
                if isinstance(v, float):
                    print(f"{v:>18.4f}", end="")
                else:
                    print(f"{v:>18}", end="")
            else:
                print(f"{'N/A':>18}", end="")
        print()

    print(f"{'─'*80}")

    # Answers
    for name in names:
        if name in results:
            print(f"\n📝 {name} Answer:")
            print(f"   {results[name].answer[:500]}")

            detail_lines: list[str] = []
            if results[name].finish_reason:
                detail_lines.append(f"finish_reason={results[name].finish_reason}")

            if detail_lines:
                print(f"   usage_details: {', '.join(detail_lines)}")

    # Token reduction metric
    if "Basic RAG" in results and "GraphRAG" in results:
        rag = results["Basic RAG"].total_tokens
        graph = results["GraphRAG"].total_tokens
        if rag > 0:
            reduction = ((rag - graph) / rag) * 100
            print(f"\n📊 Token Reduction (GraphRAG vs Basic RAG): {reduction:.1f}%")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else (
        "What compounds treat epilepsy and what are their side effects?"
    )
    run_comparison(q)
