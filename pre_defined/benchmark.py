"""Benchmark RAG, GraphRAG and Agentic GraphRAG on the Wikipedia corpus.

Every number comes from the same `PipelineResult`/`Trace`: tokens and cost from the
LLM API responses, latency from per-operation timers. Pipelines run sequentially per
question so their latencies are not skewed by contention.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pre_defined.evaluation import load_questions, score_answer
from pre_defined.category_scan import clear_graph_event_cache
from pre_defined.pipelines import DEFAULT_CONTEXT_TOKENS, DEFAULT_MODEL, DEFAULT_TOP_K, PIPELINES, RUNNERS

AGENTIC = "Agentic GraphRAG"
LLM_ONLY = "LLM-Only"


@dataclass
class BenchmarkResult:
    qid: str
    pipeline: str
    qtype: str
    question: str
    answer: str = ""
    correct: bool | None = None
    confidence: float = 0.0
    completeness: float | None = None  # gold documents present in retrieved evidence
    citation_precision: float | None = None
    citation_recall: float | None = None
    context_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    chunks: int = 0
    citations: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    strategy_changes: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    error: str = ""
    applicable: bool = True


def _gold_metrics(item: dict[str, Any], result: Any) -> dict[str, float | None]:
    gold = {str(doc_id).upper() for doc_id in item.get("gold_doc_ids", [])}
    if not gold:
        return {"completeness": None, "citation_precision": None, "citation_recall": None}
    retrieved = {doc_id.upper() for doc_id in result.retrieved_doc_ids}
    cited = {citation.upper() for citation in result.citations}
    return {
        "completeness": round(len(gold & retrieved) / len(gold), 4),
        "citation_precision": round(len(gold & cited) / len(cited), 4) if cited else None,
        "citation_recall": round(len(gold & cited) / len(gold), 4),
    }


async def run_question(
    connection: Any, item: dict[str, Any], pipeline: str, model: str,
    top_k: int, num_hops: int, num_seen_min: int, context_tokens: int,
) -> BenchmarkResult:
    row = BenchmarkResult(item["qid"], pipeline, item["qtype"], item["question"])
    started = time.perf_counter()
    clear_graph_event_cache()
    try:
        result = await RUNNERS[pipeline](
            connection, item["question"], model, top_k, num_hops, num_seen_min, context_tokens
        )
    except Exception as error:
        row.error = str(error)
        row.latency_s = round(time.perf_counter() - started, 3)
        return row
    score = score_answer(result.answer, item.get("answer", []), item["qtype"])
    row.answer = result.answer
    row.correct = score["correct"]
    row.confidence = float(score["confidence"])
    row.context_tokens = result.context_tokens
    row.prompt_tokens = result.prompt_tokens
    row.completion_tokens = result.completion_tokens
    row.total_tokens = result.total_tokens
    row.cost = result.cost
    row.latency_s = result.latency_s
    row.chunks = result.chunks
    row.citations = result.citations
    row.retrieved_doc_ids = result.retrieved_doc_ids
    row.steps = result.steps
    row.strategy_changes = result.strategy_changes
    row.stop_reason = result.stop_reason
    row.applicable = result.applicable
    if pipeline == LLM_ONLY:
        return row
    if not result.applicable:
        return row
    for name, value in _gold_metrics(item, result).items():
        setattr(row, name, value)
    return row


def _mean(values: Sequence[float | int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 4) if present else None


def _summarize_pipeline(rows: Sequence[BenchmarkResult]) -> dict[str, Any]:
    ok = [row for row in rows if not row.error and row.applicable]
    scored = [row for row in ok if row.correct is not None]
    return {
        "questions": len(rows),
        "applicable_questions": len(ok),
        "errors": sum(bool(row.error) for row in rows),
        "not_applicable": sum(not row.applicable for row in rows),
        "accuracy": _mean([float(row.correct) for row in scored]),
        "completeness": _mean([row.completeness for row in ok]),
        "citation_precision": _mean([row.citation_precision for row in ok]),
        "citation_recall": _mean([row.citation_recall for row in ok]),
        "avg_context_tokens": _mean([row.context_tokens for row in ok]),
        "avg_prompt_tokens": _mean([row.prompt_tokens for row in ok]),
        "avg_completion_tokens": _mean([row.completion_tokens for row in ok]),
        "avg_total_tokens": _mean([row.total_tokens for row in ok]),
        "avg_cost": _mean([row.cost for row in ok]),
        "avg_latency_s": _mean([row.latency_s for row in ok]),
        "avg_chunks": _mean([row.chunks for row in ok]),
        "avg_citations": _mean([len(row.citations) for row in ok]),
    }


def _summarize_agentic(rows: Sequence[BenchmarkResult]) -> dict[str, Any]:
    ok = [row for row in rows if not row.error]
    if not ok:
        return {}
    by_agent_tokens: dict[str, list[int]] = defaultdict(list)
    by_agent_latency: dict[str, list[float]] = defaultdict(list)
    tools: Counter[str] = Counter()
    for row in ok:
        per_agent_tokens: Counter[str] = Counter()
        per_agent_latency: Counter[str] = Counter()
        for step in row.steps:
            per_agent_tokens[step["agent"]] += step["total_tokens"]
            per_agent_latency[step["agent"]] += step["latency_s"]
            if step["operation"] == "retrieval":
                tools[step["tool"]] += 1
        for agent in per_agent_tokens:
            by_agent_tokens[agent].append(per_agent_tokens[agent])
            by_agent_latency[agent].append(per_agent_latency[agent])
    return {
        "avg_retrieval_steps": _mean([sum(s["operation"] == "retrieval" for s in row.steps) for row in ok]),
        "avg_llm_steps": _mean([sum(s["operation"] == "llm" for s in row.steps) for row in ok]),
        "tool_calls": dict(tools),
        "strategy_change_rate": _mean([float(bool(row.strategy_changes)) for row in ok]),
        "tool_switch_rate": _mean(
            [float(any(c["type"] == "tool_switch" for c in row.strategy_changes)) for row in ok]
        ),
        "stop_reasons": dict(Counter(row.stop_reason for row in ok)),
        "avg_tokens_by_agent": {a: _mean(v) for a, v in by_agent_tokens.items()},
        "avg_latency_by_agent_s": {a: _mean(v) for a, v in by_agent_latency.items()},
    }


def _cost_benefit(rows: Sequence[BenchmarkResult], baseline: str, challenger: str) -> dict[str, Any]:
    """Compare two pipelines on the questions both answered without error."""
    by_pipeline = {
        name: {row.qid: row for row in rows if row.pipeline == name and not row.error and row.applicable}
        for name in (baseline, challenger)
    }
    shared = sorted(set(by_pipeline[baseline]) & set(by_pipeline[challenger]))
    if not shared:
        return {}
    base = [by_pipeline[baseline][qid] for qid in shared]
    chal = [by_pipeline[challenger][qid] for qid in shared]
    extra_tokens = sum(r.total_tokens for r in chal) - sum(r.total_tokens for r in base)
    extra_correct = sum(bool(r.correct) for r in chal) - sum(bool(r.correct) for r in base)
    scored = [r for r in base if r.correct is not None]
    return {
        "questions_compared": len(shared),
        "extra_correct_answers": extra_correct if scored else None,
        "token_multiple": round(sum(r.total_tokens for r in chal) / max(1, sum(r.total_tokens for r in base)), 3),
        "extra_cost": round(sum(r.cost for r in chal) - sum(r.cost for r in base), 6),
        "extra_latency_s": round(sum(r.latency_s for r in chal) - sum(r.latency_s for r in base), 3),
        "extra_tokens_per_extra_correct": round(extra_tokens / extra_correct, 1) if scored and extra_correct > 0 else None,
        "wins": sum(bool(c.correct) and not b.correct for b, c in zip(base, chal)),
        "losses": sum(bool(b.correct) and not c.correct for b, c in zip(base, chal)),
    }


def summarize(results: Sequence[BenchmarkResult], pipelines: Sequence[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pipelines": {name: _summarize_pipeline([r for r in results if r.pipeline == name]) for name in pipelines},
    }
    if AGENTIC in pipelines:
        summary["agentic_behavior"] = _summarize_agentic([r for r in results if r.pipeline == AGENTIC])
        summary["agentic_vs"] = {
            name: _cost_benefit(results, name, AGENTIC) for name in pipelines if name != AGENTIC
        }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    def fmt(value: float | None, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    print(f"\n{'Pipeline':<18} {'Acc':>7} {'Compl':>7} {'CtxTok':>8} {'Prompt':>8} {'Compl.':>8} {'Total':>8} {'Cost':>10} {'Lat(s)':>8}")
    for name, m in summary["pipelines"].items():
        print(
            f"{name:<18} {fmt(m['accuracy'], '.1%'):>7} {fmt(m['completeness'], '.1%'):>7} "
            f"{fmt(m['avg_context_tokens'], '.0f'):>8} {fmt(m['avg_prompt_tokens'], '.0f'):>8} "
            f"{fmt(m['avg_completion_tokens'], '.0f'):>8} {fmt(m['avg_total_tokens'], '.0f'):>8} "
            f"{fmt(m['avg_cost'], '.6f'):>10} {fmt(m['avg_latency_s'], '.2f'):>8}"
        )
    behavior = summary.get("agentic_behavior")
    if behavior:
        print("\nAgentic behavior")
        for key, value in behavior.items():
            print(f"  {key}: {value}")
        for name, comparison in summary["agentic_vs"].items():
            print(f"\nAgentic vs {name}: {comparison}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RAG, GraphRAG and Agentic GraphRAG")
    parser.add_argument("--dataset", choices=("eval_public", "eval_hidden"), default="eval_public")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=os.getenv("LLM_COMPLETION_MODEL", DEFAULT_MODEL))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--num-seen-min", type=int, default=1)
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    parser.add_argument("--pipeline", choices=PIPELINES, action="append", dest="pipelines")
    parser.add_argument("--output", type=Path, default=Path("pre_defined/data/benchmark_results.json"))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")
    if args.top_k < 1 or args.num_hops < 0 or args.num_seen_min < 1 or args.context_tokens < 1:
        parser.error("top-k, num-seen-min and context-tokens must be positive; num-hops cannot be negative")
    return args


async def _run(args: argparse.Namespace) -> tuple[list[BenchmarkResult], dict[str, Any]]:
    from pre_defined.main import _connection

    pipelines = tuple(args.pipelines or PIPELINES)
    connection = _connection()
    results: list[BenchmarkResult] = []
    for item in load_questions(args.dataset, args.limit):
        for pipeline in pipelines:
            results.append(await run_question(
                connection, item, pipeline, args.model,
                args.top_k, args.num_hops, args.num_seen_min, args.context_tokens,
            ))
    summary = summarize(results, pipelines)
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "results": [asdict(result) for result in results],
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results, summary


def main() -> None:
    args = _parse_args()
    _, summary = asyncio.run(_run(args))
    _print_summary(summary)
    print(f"\nDetailed results: {args.output}")


if __name__ == "__main__":
    main()
