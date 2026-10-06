"""RAG, GraphRAG and Agentic GraphRAG over the Savanna-backed Wikipedia corpus."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from llm.ag_llm import LLMClient, PipelineResult
from pre_defined.category_scan import (
    clear_graph_event_cache, requires_event_scan, scan_category, scan_venue_date,
)
from pre_defined.prompts import (
    EVALUATOR_SYSTEM, LLM_ONLY_SYSTEM, PLANNER_SYSTEM, QUERY_REWRITE_SYSTEM, answer_messages,
)
from pre_defined.retrieval import Chunk, Evidence, Retrieval, search
from pre_defined.trace import Trace, count_tokens

PIPELINES = ("LLM-Only", "RAG", "Agentic GraphRAG")  # "Structured Graph"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_CONTEXT_TOKENS = 6000
EVALUATOR_CONTEXT_TOKENS = 3000
DEFAULT_TOP_K = 15
MAX_AGENT_STEPS = 4
TOOLS = {
    "similarity_search": "similarity",
    "graph_search": "hybrid",
    "community_search": "community",
    "event_category_scan": "event_category_scan",
}
GRAPH_TOOLS = {"graph_search", "community_search", "event_category_scan"}
_RELATIVE_REFERENCE_RE = re.compile(
    r"\b(?:before|after|preceding|following|previous|prior to|next|latest|most recent|earliest)\b", re.IGNORECASE
)


def _parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _citations(answer: str, evidence: Evidence) -> list[str]:
    cited = {
        part.strip().upper()
        for group in re.findall(r"\[([^\[\]]+)\]", answer)
        for part in group.split(",")
    }
    return sorted(cited & {source.upper() for source in evidence.source_ids})


def _retrieval_tokens(retrieval: Retrieval, model: str) -> int:
    single = Evidence()
    single.add(retrieval)
    return count_tokens(single.render(model, 10**9), model)


async def _retrieve(
    connection: Any, query: str, method: str, top_k: int, num_hops: int, num_seen_min: int
) -> Retrieval:
    return await asyncio.to_thread(search, connection, query, method, top_k, num_hops, num_seen_min)


def _record_retrieval(
    trace: Trace, agent: str, tool: str, retrieval: Retrieval, new_chunks: int, model: str
) -> None:
    chunks = sum(1 for chunk in retrieval.chunks if chunk.kind != "entity")
    trace.retrieval(
        agent, tool, retrieval.query, retrieval.latency_s,
        _retrieval_tokens(retrieval, model), chunks, new_chunks,
    )


async def _answer(
    llm: LLMClient, trace: Trace, question: str, evidence: Evidence,
    model: str, context_tokens: int, graph: bool, agent: str,
) -> tuple[PipelineResult, str]:
    context = evidence.render(model, context_tokens, track=True)
    result = await llm.query(answer_messages(question, context, graph))
    trace.llm(agent, result, detail=f"context={count_tokens(context, model)} tokens")
    return result, context


def _build_result(
    trace: Trace, answer: PipelineResult, evidence: Evidence, context: str,
    model: str, started: float, graph_used: bool,
) -> PipelineResult:
    return PipelineResult(
        answer=answer.answer,
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
        total_tokens=trace.total_tokens,
        latency_s=round(time.perf_counter() - started, 3),
        cost=trace.cost,
        model=model,
        retrieved_context=context,
        retrieved_doc_ids=evidence.doc_ids,
        graph_used=graph_used,
        finish_reason=answer.finish_reason,
        context_tokens=count_tokens(context, model),
        chunks=evidence.chunk_count,
        citations=_citations(answer.answer, evidence),
        steps=trace.as_dicts(),
        strategy_changes=trace.strategy_changes,
        stop_reason=trace.stop_reason,
        applicable=answer.applicable,
    )


async def _structured_graph_pipeline(
    connection: Any, question: str, model: str, started: float,
) -> PipelineResult | None:
    scan = await asyncio.to_thread(scan_category, connection, question)
    if scan is None:
        scan = await asyncio.to_thread(scan_venue_date, connection, question)
    if scan is None:
        return None

    evidence = Evidence()
    retrieval = Retrieval(
        method="category_scan",
        query=question,
        chunks=[
            Chunk(event.doc_id.lower(), event.doc_id, line, "chunk")
            for event, line in zip(scan.events, scan.context.splitlines())
        ],
        relationships=[],
        latency_s=scan.latency_s,
    )
    trace = Trace()
    _record_retrieval(trace, "category_retriever", "category_scan", retrieval, evidence.add(retrieval), model)
    evidence.shown_sources = set(scan.doc_ids)
    trace.stop_reason = scan.stop_reason
    answer = scan.answer if scan.complete else "Final answer: Insufficient evidence; " + scan.stop_reason
    result = PipelineResult(answer=answer)
    return _build_result(trace, result, evidence, scan.context, model, started, graph_used=True)


async def run_structured_graph(
    connection: Any, question: str, model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K,
    num_hops: int = 2, num_seen_min: int = 1, context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> PipelineResult:
    """Exact graph-field baseline; no LLM answer generation."""
    started = time.perf_counter()
    result = await _structured_graph_pipeline(connection, question, model, started)
    if result is not None:
        return result
    return PipelineResult(
        answer="",
        model=model,
        latency_s=round(time.perf_counter() - started, 3),
        stop_reason="not applicable: requires category or unique venue/date question",
        applicable=False,
    )


async def _event_scan_retrieval(connection: Any, question: str) -> Retrieval | None:
    scan = await asyncio.to_thread(scan_category, connection, question)
    if scan is None:
        scan = await asyncio.to_thread(scan_venue_date, connection, question)
    if scan is None:
        return None
    context = scan.context
    if scan.complete:
        context = f"Graph coverage: complete. {scan.stop_reason}\n\n{context}"
    else:
        context = f"Graph coverage: incomplete. {scan.stop_reason}\n\n{context}"
    chunks = [
        Chunk(event.doc_id.lower(), event.doc_id, line, "chunk")
        for event, line in zip(scan.events, scan.context.splitlines())
    ]
    chunks.append(Chunk("graph_scan_status", "graph_scan_status", context.split("\n\n", 1)[0], "entity"))
    return Retrieval("event_category_scan", question, chunks, [], scan.latency_s)


async def _search_query(llm: LLMClient, trace: Trace, question: str) -> str:
    """Resolve relative references ("immediately before 2020") into explicit years; skip the LLM call otherwise."""
    if not _RELATIVE_REFERENCE_RE.search(question):
        return question
    result = await llm.query([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": question},
    ])
    query = result.answer.strip().strip('"')
    trace.llm("query_rewriter", result, detail=query)
    return query or question


async def _single_retrieval_pipeline(
    connection: Any, question: str, model: str, method: str, top_k: int,
    num_hops: int, num_seen_min: int, context_tokens: int, graph: bool,
) -> PipelineResult:
    started = time.perf_counter()
    trace, evidence, llm = Trace(), Evidence(), LLMClient(model=model)
    query = await _search_query(llm, trace, question)
    retrieval = await _retrieve(connection, query, method, top_k, num_hops, num_seen_min)
    _record_retrieval(trace, "retriever", f"{method}_search", retrieval, evidence.add(retrieval), model)
    answer, context = await _answer(llm, trace, question, evidence, model, context_tokens, graph, "generator")
    trace.stop_reason = "single retrieval pass"
    return _build_result(trace, answer, evidence, context, model, started, graph_used=graph)


async def run_llm_only(
    connection: Any, question: str, model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K,
    num_hops: int = 2, num_seen_min: int = 1, context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> PipelineResult:
    """No retrieval: the question goes straight to the model, which answers from training knowledge."""
    started = time.perf_counter()
    trace = Trace()
    result = await LLMClient(model=model).query([
        {"role": "system", "content": LLM_ONLY_SYSTEM},
        {"role": "user", "content": question},
    ])
    trace.llm("generator", result, detail="no retrieval")
    trace.stop_reason = "no retrieval; parametric answer"
    return _build_result(trace, result, Evidence(), "", model, started, graph_used=False)


async def run_rag(
    connection: Any, question: str, model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K,
    num_hops: int = 2, num_seen_min: int = 1, context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> PipelineResult:
    return await _single_retrieval_pipeline(
        connection, question, model, "similarity", top_k, num_hops, num_seen_min, context_tokens, graph=False
    )


async def run_graph_rag(
    connection: Any, question: str, model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K,
    num_hops: int = 2, num_seen_min: int = 1, context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> PipelineResult:
    # Hybrid alone returns hundreds of entity rows and few document chunks, so seed with similarity hits.
    started = time.perf_counter()
    trace, evidence, llm = Trace(), Evidence(), LLMClient(model=model)
    query = await _search_query(llm, trace, question)
    similar, hybrid = await asyncio.gather(
        _retrieve(connection, query, "similarity", top_k, num_hops, num_seen_min),
        _retrieve(connection, query, "hybrid", top_k, num_hops, num_seen_min),
    )
    for retrieval, tool in ((similar, "similarity_search"), (hybrid, "hybrid_search")):
        _record_retrieval(trace, "retriever", tool, retrieval, evidence.add(retrieval), model)
    answer, context = await _answer(llm, trace, question, evidence, model, context_tokens, True, "generator")
    trace.stop_reason = "single retrieval pass"
    return _build_result(trace, answer, evidence, context, model, started, graph_used=True)


def _valid_action(value: dict[str, Any]) -> tuple[str, str] | None:
    tool, query = value.get("tool"), value.get("query")
    if tool in TOOLS and isinstance(query, str) and query.strip():
        return tool, query.strip()
    return None


def _untried_tool(tried: set[tuple[str, str]], query: str) -> tuple[str, str] | None:
    return next(((tool, query) for tool in TOOLS if (tool, query) not in tried), None)


async def _plan(llm: LLMClient, trace: Trace, question: str) -> tuple[str, str]:
    result = await llm.query([
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ])
    plan = _parse_json(result.answer)
    action = _valid_action(plan)
    trace.llm("planner", result, detail=f"{action[0] if action else 'fallback'}: {plan.get('reason', '')}")
    return action or ("similarity_search", question)


async def _evaluate(
    llm: LLMClient, trace: Trace, question: str, evidence: Evidence, history: list[str], model: str,
) -> dict[str, Any]:
    prompt = (
        f"Question: {question}\n\nRetrievals so far:\n" + "\n".join(history)
        + f"\n\nEvidence:\n{evidence.render(model, EVALUATOR_CONTEXT_TOKENS)}"
    )
    result = await llm.query([
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    verdict = _parse_json(result.answer)
    trace.llm(
        "evaluator", result,
        detail=f"sufficient={verdict.get('sufficient')}; missing={verdict.get('missing') or '-'}",
    )
    return verdict


async def _decide_next(
    llm: LLMClient, trace: Trace, question: str, evidence: Evidence, history: list[str], model: str,
    tried: set[tuple[str, str]], query: str, new_chunks: int,
) -> tuple[tuple[str, str] | None, str, str]:
    """Return (next action, why, stop reason); a stop reason with no action ends the investigation."""
    if new_chunks == 0:
        # Switch tool on the same query before giving up.
        return _untried_tool(tried, query), "latest retrieval added nothing new", "no new evidence and every tool already tried"
    verdict = await _evaluate(llm, trace, question, evidence, history, model)
    if verdict.get("sufficient") is True:
        return None, "", "evaluator: evidence sufficient"
    action = _valid_action(verdict)
    if action is None or action in tried:
        action = _untried_tool(tried, query)
    why = str(verdict.get("missing") or verdict.get("reason") or "")
    return action, why, "evaluator proposed no new retrieval and every tool already tried"


async def _run_agent_tool(
    connection: Any, tool: str, query: str, top_k: int, num_hops: int, num_seen_min: int,
) -> tuple[Retrieval, str]:
    if tool == "event_category_scan":
        retrieval = await _event_scan_retrieval(connection, query)
        if retrieval is not None:
            return retrieval, "graph event scan"
        tool = "similarity_search"
    retrieval = await _retrieve(connection, query, TOOLS[tool], top_k, num_hops, num_seen_min)
    label = "similarity_search_fallback" if tool == "similarity_search" else tool
    return retrieval, label


async def run_agentic(
    connection: Any, question: str, model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K,
    num_hops: int = 2, num_seen_min: int = 1, context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    max_steps: int = MAX_AGENT_STEPS,
) -> PipelineResult:
    started = time.perf_counter()
    trace, evidence, llm = Trace(), Evidence(), LLMClient(model=model)
    tool, query = await _plan(llm, trace, question)
    if requires_event_scan(question) and (tool != "event_category_scan" or query != question):
        trace.change_strategy(
            "coverage_policy",
            f"{tool}: {query}",
            f"event_category_scan: {question}",
            "question requires complete category or unique venue/date coverage; preserving the full question",
        )
        tool, query = "event_category_scan", question
    tried: set[tuple[str, str]] = set()
    history: list[str] = []

    for attempt in range(1, max_steps + 1):
        tried.add((tool, query))
        retrieval, tool_label = await _run_agent_tool(
            connection, tool, query, top_k, num_hops, num_seen_min
        )
        new_chunks = evidence.add(retrieval)
        _record_retrieval(trace, f"{tool.split('_')[0]}_retriever", tool_label, retrieval, new_chunks, model)
        history.append(f'{attempt}. {tool} "{query}" -> {new_chunks} new chunks')

        if attempt == max_steps:
            trace.stop_reason = "step budget exhausted"
            break

        action, why, stop_reason = await _decide_next(
            llm, trace, question, evidence, history, model, tried, query, new_chunks
        )
        if action is None:
            trace.stop_reason = stop_reason
            break
        if action[0] != tool:
            trace.change_strategy("tool_switch", tool, action[0], why)
        else:
            trace.change_strategy("query_refinement", query, action[1], why)
        tool, query = action

    answer, context = await _answer(
        llm, trace, question, evidence, model, context_tokens, graph=True, agent="synthesizer"
    )
    used_graph = any(step.tool in GRAPH_TOOLS for step in trace.steps)
    return _build_result(trace, answer, evidence, context, model, started, graph_used=used_graph)


RUNNERS = {
    "LLM-Only": run_llm_only,
    "RAG": run_rag,
    "GraphRAG": run_graph_rag,
    "Agentic GraphRAG": run_agentic,
    # "Structured Graph": run_structured_graph,
}


async def run_all(
    connection: Any, question: str, model: str, top_k: int, num_hops: int, num_seen_min: int,
) -> tuple[dict[str, PipelineResult], dict[str, Exception]]:
    clear_graph_event_cache()
    outcomes = await asyncio.gather(
        *(RUNNERS[name](connection, question, model, top_k, num_hops, num_seen_min) for name in PIPELINES),
        return_exceptions=True,
    )
    results: dict[str, PipelineResult] = {}
    errors: dict[str, Exception] = {}
    for name, outcome in zip(PIPELINES, outcomes):
        if isinstance(outcome, Exception):
            errors[name] = outcome
        elif isinstance(outcome, BaseException):
            raise outcome
        else:
            results[name] = outcome
    return results, errors
