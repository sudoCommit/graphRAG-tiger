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

    results: dict[str, PipelineResult] = {}

    # Pipeline 1: LLM-Only
    print("▶ Running Pipeline 1: LLM-Only...")
    try:
        results["LLM-Only"] = pipeline1.query(question, model=model)
        print(f"  ✅ Done ({results['LLM-Only'].latency_s}s)")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Pipeline 2: Basic RAG (vector-only)
    print("▶ Running Pipeline 2: Basic RAG (vector-only)...")
    try:
        results["Basic RAG"] = pipeline2.query(question, model=model)
        print(f"  ✅ Done ({results['Basic RAG'].latency_s}s)")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Pipeline 3: GraphRAG (hybrid)
    print("▶ Running Pipeline 3: GraphRAG (hybrid)...")
    try:
        results["GraphRAG"] = pipeline3.query(question, model=model)
        print(f"  ✅ Done ({results['GraphRAG'].latency_s}s)")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Comparison table
    _print_comparison(results)
    return results


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
