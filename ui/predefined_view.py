"""Pre-defined Wikipedia corpus section: single query, 3-pipeline comparison, and evaluation."""

import asyncio

import pandas as pd
import streamlit as st

from llm.ag_llm import PipelineResult
from pre_defined.evaluation import load_questions, score_answer
from pre_defined.main import _connection, query_graph
from pre_defined.pipelines import DEFAULT_TOP_K, PIPELINES as PREDEFINED_PIPELINES, run_all as run_all_predefined
from ui.config import EMPTY_QUESTION_WARNING, MODEL_OPTIONS, PIPELINE_META, PREDEFINED_SECTION
from ui.views import render_context, render_pipeline_slot


CORRECT_METRIC = "Correct"
TOTAL_TOKENS = "Total Tokens"
PROMPT_TOKENS = "Prompt Tokens"
COMPLETION_TOKENS = "Completion Tokens"
CONTEXT_TOKENS = "Context Tokens"
COST_METRIC = "Cost ($)"
RECALL_METRIC = "Recall"
LATENCY_METRIC = "Latency (s)"
STEPS_METRIC = "Steps"
CHUNKS_METRIC = "Chunks"
CITATIONS_METRIC = "Citations"
STOP_METRIC = "Stop reason"
PIPELINE_COL = "Pipeline"
ACCURACY_COL = "Accuracy (%)"
AVG_TOKENS_COL = "Avg tokens"
TOTAL_COST_COL = "Total cost ($)"
BASELINE_PIPELINE = "GraphRAG"
AGENTIC_PIPELINE = "Agentic GraphRAG"
STRUCTURED_PIPELINE = "Structured Graph"


def _render_predefined_query_tab(method: str, top_k: int, num_hops: int, num_seen_min: int) -> None:
    question = st.text_area(
        "Question",
        height=120,
        placeholder="e.g. Who won the gold medal in the men's 20 kilometres walk at the 2012 Olympics?",
        key="predefined_question",
    )
    if st.button("🚀 Ask predefined corpus", type="primary", key="predefined_run"):
        if not question.strip():
            st.warning(EMPTY_QUESTION_WARNING)
            return
        try:
            with st.spinner("Querying the predefined corpus..."):
                answer = query_graph(
                    _connection(), question, method, top_k, num_hops, num_seen_min
                )
            st.session_state["predefined_answer"] = answer
            st.session_state.pop("predefined_error", None)
        except Exception as error:
            st.session_state["predefined_error"] = error
            st.session_state.pop("predefined_answer", None)

    if "predefined_error" in st.session_state:
        st.error(f"Query failed: {st.session_state['predefined_error']}")
    elif "predefined_answer" in st.session_state:
        st.markdown("#### Answer")
        st.write(st.session_state["predefined_answer"])


def _recall(doc_ids: object, gold_doc_ids: object) -> float | None:
    gold = {str(identifier) for identifier in (gold_doc_ids or [])}
    if not gold:
        return None
    retrieved = {str(identifier) for identifier in (doc_ids or [])}
    return round(len(gold & retrieved) / len(gold), 4)


def _pipeline_records(item: dict, results: dict, errors: dict) -> list[dict[str, object]]:
    gold_answers = item.get("answer", [])
    records: list[dict[str, object]] = []
    for name in PREDEFINED_PIPELINES:
        base = {"QID": item["qid"], "Type": item["qtype"], PIPELINE_COL: name}
        if name in errors:
            records.append({
                **base, "Status": f"error: {errors[name]}", CORRECT_METRIC: None,
                "Confidence": None, TOTAL_TOKENS: 0, COST_METRIC: 0.0,
                LATENCY_METRIC: None, RECALL_METRIC: None, "Answer": "",
            })
            continue
        result = results[name]
        if not result.applicable:
            records.append({
                **base, "Status": "not applicable", CORRECT_METRIC: None,
                "Confidence": None, TOTAL_TOKENS: 0, COST_METRIC: 0.0,
                LATENCY_METRIC: result.latency_s, RECALL_METRIC: None,
                "Answer": "",
            })
            continue
        score = score_answer(result.answer, gold_answers, item["qtype"])
        records.append({
            **base, "Status": "ok", CORRECT_METRIC: score["correct"],
            "Confidence": score["confidence"],
            PROMPT_TOKENS: result.prompt_tokens, COMPLETION_TOKENS: result.completion_tokens,
            TOTAL_TOKENS: result.total_tokens, CONTEXT_TOKENS: result.context_tokens,
            COST_METRIC: round(result.cost, 6), LATENCY_METRIC: result.latency_s,
            RECALL_METRIC: None if name == "LLM-Only" else _recall(result.retrieved_doc_ids, item.get("gold_doc_ids")),
            STEPS_METRIC: len(result.steps), CHUNKS_METRIC: result.chunks,
            CITATIONS_METRIC: len(result.citations), STOP_METRIC: result.stop_reason,
            "Answer": result.answer,
        })
    return records


def _pipeline_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame[frame["Status"] == "ok"]
    rows = []
    for name in PREDEFINED_PIPELINES:
        sub = ok[ok[PIPELINE_COL] == name]
        if sub.empty:
            continue
        scored = sub[sub[CORRECT_METRIC].notna()]
        recalls = sub[RECALL_METRIC].dropna()
        latencies = sub[LATENCY_METRIC].dropna()
        rows.append({
            PIPELINE_COL: name,
            "Executed": len(sub),
            "Not applicable": int((frame[(frame[PIPELINE_COL] == name) & (frame["Status"] == "not applicable")]).shape[0]),
            ACCURACY_COL: round(scored[CORRECT_METRIC].mean() * 100, 1) if len(scored) else None,
            "Avg confidence (%)": round(scored["Confidence"].mean() * 100, 1) if len(scored) else None,
            "Avg recall (%)": round(recalls.mean() * 100, 1) if len(recalls) else None,
            "Avg context tokens": round(sub[CONTEXT_TOKENS].mean(), 0),
            "Avg prompt tokens": round(sub[PROMPT_TOKENS].mean(), 0),
            "Avg completion tokens": round(sub[COMPLETION_TOKENS].mean(), 0),
            AVG_TOKENS_COL: round(sub[TOTAL_TOKENS].mean(), 0),
            TOTAL_COST_COL: round(sub[COST_METRIC].sum(), 6),
            "Avg latency (s)": round(latencies.mean(), 3) if len(latencies) else None,
            "Avg steps": round(sub[STEPS_METRIC].mean(), 2),
            "Avg chunks": round(sub[CHUNKS_METRIC].mean(), 1),
            "Avg citations": round(sub[CITATIONS_METRIC].mean(), 1),
        })
    return pd.DataFrame(rows).set_index(PIPELINE_COL) if rows else pd.DataFrame()


def _render_agentic_verdict(summary: pd.DataFrame, frame: pd.DataFrame) -> None:
    if STRUCTURED_PIPELINE in summary.index:
        st.caption("Structured Graph is reported only for supported complete graph scans; it does not call an LLM.")
    if not {BASELINE_PIPELINE, AGENTIC_PIPELINE}.issubset(summary.index):
        return
    base, agent = summary.loc[BASELINE_PIPELINE], summary.loc[AGENTIC_PIPELINE]
    token_multiple = agent[AVG_TOKENS_COL] / base[AVG_TOKENS_COL] if base[AVG_TOKENS_COL] else None
    line = (
        f"Agentic GraphRAG uses {token_multiple:.2f}x the tokens and "
        f"{agent['Avg latency (s)'] - base['Avg latency (s)']:+.1f}s latency of GraphRAG"
        if token_multiple else "Agentic GraphRAG vs GraphRAG"
    )
    if pd.notna(agent[ACCURACY_COL]) and pd.notna(base[ACCURACY_COL]):
        gain = (agent[ACCURACY_COL] - base[ACCURACY_COL]) / 100
        line += f" for {gain * 100:+.1f} pp accuracy"
        if gain > 0:
            extra = agent[AVG_TOKENS_COL] - base[AVG_TOKENS_COL]
            line += f" (~{extra / gain:,.0f} extra tokens per additional correct answer)"
    st.caption(line + ".")
    agentic = frame[(frame[PIPELINE_COL] == AGENTIC_PIPELINE) & (frame["Status"] == "ok")]
    if len(agentic):
        stops = agentic[STOP_METRIC].value_counts().to_dict()
        st.caption(f"Agentic stop reasons: {stops}")


def _render_eval_comparison(records: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(records)
    failed = frame[frame["Status"].str.startswith("error")]
    for name, group in failed.groupby(PIPELINE_COL):
        st.error(f"{name} failed on {len(group)} question(s). First error: {group['Status'].iloc[0]}")
    summary = _pipeline_summary(frame)
    if summary.empty:
        st.warning("No successful pipeline runs to compare.")
        return

    st.markdown('<div class="section-label">Pipeline comparison</div>', unsafe_allow_html=True)
    cards = st.columns(len(summary.index))
    for column, name in zip(cards, summary.index):
        accuracy = summary.loc[name, ACCURACY_COL]
        tokens = summary.loc[name, AVG_TOKENS_COL]
        latency = summary.loc[name, "Avg latency (s)"]
        column.metric(
            name,
            f"{accuracy:.1f}%" if pd.notna(accuracy) else f"{latency:.1f}s",
            help=f"Avg {tokens:,.0f} tokens · ${summary.loc[name, TOTAL_COST_COL]:.6f} total"
            + ("" if pd.notna(accuracy) else " · accuracy unavailable (no gold answers); showing avg latency"),
        )

    st.dataframe(
        summary,
        width="stretch",
        column_config={
            ACCURACY_COL: st.column_config.NumberColumn(format="%.1f%%"),
            "Avg confidence (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Avg recall (%)": st.column_config.NumberColumn(format="%.1f%%"),
            AVG_TOKENS_COL: st.column_config.NumberColumn(format="%d"),
            TOTAL_COST_COL: st.column_config.NumberColumn(format="$%.6f"),
            "Avg latency (s)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    _render_agentic_verdict(summary, frame)

    charts = st.columns(3)
    with charts[0]:
        if summary[ACCURACY_COL].notna().any():
            st.caption("Accuracy by pipeline (%)")
            st.bar_chart(summary[ACCURACY_COL], height=240, color="#2a9d8f")
        else:
            st.caption("Avg latency by pipeline (s)")
            st.bar_chart(summary["Avg latency (s)"], height=240, color="#2a9d8f")
    with charts[1]:
        st.caption("Avg tokens by pipeline")
        st.bar_chart(summary[AVG_TOKENS_COL], height=240, color="#fda367")
    with charts[2]:
        st.caption("Total cost by pipeline ($)")
        st.bar_chart(summary[TOTAL_COST_COL], height=240, color="#5dd6e6")

    detail_columns = [
        "QID", "Type", PIPELINE_COL, CORRECT_METRIC, "Confidence", RECALL_METRIC,
        CONTEXT_TOKENS, PROMPT_TOKENS, COMPLETION_TOKENS, TOTAL_TOKENS, COST_METRIC,
        LATENCY_METRIC, STEPS_METRIC, CHUNKS_METRIC, CITATIONS_METRIC, STOP_METRIC,
        "Answer", "Status",
    ]
    detail = frame.reindex(columns=detail_columns)
    st.markdown('<div class="section-label">Per-question detail</div>', unsafe_allow_html=True)
    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
        column_config={
            CORRECT_METRIC: st.column_config.CheckboxColumn("Correct"),
            "Confidence": st.column_config.NumberColumn(format="%.2f"),
            RECALL_METRIC: st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
            TOTAL_TOKENS: st.column_config.NumberColumn(format="%d"),
            COST_METRIC: st.column_config.NumberColumn(format="$%.6f"),
            LATENCY_METRIC: st.column_config.NumberColumn(format="%.2f"),
            "Answer": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "⬇️ Download comparison (CSV)",
        data=detail.to_csv(index=False),
        file_name="evaluation_comparison.csv",
        mime="text/csv",
        key="predefined_eval_download",
    )


def _render_evaluation_tab(model: str, top_k: int, num_hops: int, num_seen_min: int) -> None:
    controls = st.columns([2, 3])
    with controls[0]:
        dataset = st.radio(
            "Question set",
            ("eval_public", "eval_hidden"),
            format_func=lambda value: "Public (scored)" if value == "eval_public" else "Hidden (execution only)",
            horizontal=True,
            key="predefined_eval_dataset",
        )
    available = load_questions(dataset, limit=None)
    with controls[1]:
        count = st.slider(
            "Questions to evaluate", 1, len(available), min(10, len(available)),
            key="predefined_eval_count",
        )
    questions = available[:count]
    st.caption(
        f"{len(questions)} of {len(available)} questions · each runs through "
        f"{', '.join(PREDEFINED_PIPELINES)}."
    )
    if st.button("▶️ Run comparison", type="primary", key="predefined_eval_run"):
        connection = _connection()
        records: list[dict[str, object]] = []
        progress = st.progress(0, text="Starting comparison...")
        for index, item in enumerate(questions, start=1):
            results, errors = asyncio.run(
                run_all_predefined(connection, item["question"], model, top_k, num_hops, num_seen_min)
            )
            records.extend(_pipeline_records(item, results, errors))
            progress.progress(index / len(questions), text=f"Completed {index} of {len(questions)}")
        progress.empty()
        st.session_state["predefined_eval_records"] = records

    records = st.session_state.get("predefined_eval_records")
    if records:
        _render_eval_comparison(records)


def _render_trace(result: PipelineResult) -> None:
    steps = pd.DataFrame(result.steps)
    if steps.empty:
        return
    columns = [
        "index", "agent", "operation", "tool", "detail", "latency_s", "prompt_tokens",
        "completion_tokens", "total_tokens", "cost", "context_tokens", "chunks", "new_chunks",
    ]
    st.dataframe(steps.reindex(columns=columns), width="stretch", hide_index=True)
    st.caption(
        f"Stop reason: {result.stop_reason} · {len(result.citations)} citation(s) · "
        f"{result.chunks} chunk(s) · strategy changes: {len(result.strategy_changes)}"
    )
    for change in result.strategy_changes:
        st.caption(f"Step {change['step']}: {change['type']} {change['from']!r} -> {change['to']!r} ({change['why']})")


def _render_compare_metrics(results: dict[str, PipelineResult]) -> None:
    frame = pd.DataFrame([
        {
            PIPELINE_COL: name,
            CONTEXT_TOKENS: result.context_tokens,
            PROMPT_TOKENS: result.prompt_tokens,
            COMPLETION_TOKENS: result.completion_tokens,
            TOTAL_TOKENS: result.total_tokens,
            COST_METRIC: result.cost,
            LATENCY_METRIC: result.latency_s,
            STEPS_METRIC: len(result.steps),
            CHUNKS_METRIC: result.chunks,
            CITATIONS_METRIC: len(result.citations),
        }
        for name, result in results.items()
    ]).set_index(PIPELINE_COL)
    st.dataframe(frame.style.format({COST_METRIC: "{:.6f}", LATENCY_METRIC: "{:.2f}"}), width="stretch")


def _slot_status(name: str, results: dict, errors: dict) -> str:
    if name in errors:
        return "error"
    return "done" if name in results else "idle"


def _render_predefined_compare_tab(model: str, top_k: int, num_hops: int, num_seen_min: int) -> None:
    st.caption(
        "LLM-Only is parametric; RAG and GraphRAG retrieve then generate; Agentic GraphRAG plans and evaluates retrieval; "
        "Structured Graph is an exact graph-field baseline for category and venue/date questions."
    )
    question = st.text_area(
        "Question",
        height=120,
        placeholder="e.g. Who won the men's 20 kilometres walk at the 2012 Summer Olympics?",
        key="predefined_compare_question",
    )
    if st.button("🚀 Run all pipelines", type="primary", key="predefined_compare_run"):
        if not question.strip():
            st.warning(EMPTY_QUESTION_WARNING)
        else:
            with st.spinner("Running the selected pipelines..."):
                results, errors = asyncio.run(
                    run_all_predefined(_connection(), question, model, top_k, num_hops, num_seen_min)
                )
            st.session_state["predefined_compare_results"] = results
            st.session_state["predefined_compare_errors"] = errors

    results = st.session_state.get("predefined_compare_results", {})
    errors = st.session_state.get("predefined_compare_errors", {})
    if not results and not errors:
        st.info("Enter a question and run to compare the pipelines.")
        return
    st.divider()
    for name, column in zip(PREDEFINED_PIPELINES, st.columns(len(PREDEFINED_PIPELINES))):
        render_pipeline_slot(column.empty(), name, _slot_status(name, results, errors), results.get(name), errors.get(name))
    if results:
        st.space()
        metrics_tab, trace_tab, context_tab = st.tabs(["📊 Metrics", "🧭 Trace", "🔍 Retrieved Context"])
        with metrics_tab:
            _render_compare_metrics(results)
        with trace_tab:
            for name, result in results.items():
                st.markdown(f"##### {name}")
                _render_trace(result)
        with context_tab:
            render_context(results)


def render_predefined_section() -> None:
    st.markdown(f'<div class="section-label">{PREDEFINED_SECTION}</div>', unsafe_allow_html=True)
    st.subheader("Wikipedia GraphRAG")
    st.caption("Ask questions against the pre-loaded Wikipedia corpus using TigerGraph GraphRAG.")

    with st.sidebar:
        st.header(PREDEFINED_SECTION, icon="📚")
        model = st.selectbox(
            "Chat model",
            MODEL_OPTIONS,
            help="Model used for LLM-Only and for token/cost pricing.",
            key="predefined_model",
        )
        method = st.selectbox(
            "Retrieval method",
            ("hybrid", "similarity", "community"),
            format_func=str.title,
            help="Used by the Normal query and Evaluation tabs.",
            key="predefined_method",
        )
        top_k = st.slider("Top-K", 1, 20, DEFAULT_TOP_K, key="predefined_top_k")
        num_hops = st.slider("Graph hops", 0, 10, 2, key="predefined_num_hops")
        num_seen_min = st.slider("Minimum paths", 1, 5, 1, key="predefined_num_seen_min")

    compare_tab, query_tab, evaluation_tab = st.tabs(
        ["⚖️ Compare pipelines", "💬 Normal query", "📈 Evaluation"]
    )
    with compare_tab:
        _render_predefined_compare_tab(model, top_k, num_hops, num_seen_min)
    with query_tab:
        _render_predefined_query_tab(method, top_k, num_hops, num_seen_min)
    with evaluation_tab:
        _render_evaluation_tab(model, top_k, num_hops, num_seen_min)
