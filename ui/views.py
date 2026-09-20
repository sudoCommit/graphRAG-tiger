import html

import altair as alt
import pandas as pd
import streamlit as st

from llm.ag_llm import PipelineResult
from ui.config import PIPELINES, PIPELINE_META


METRIC_LATENCY = "Latency (s)"
METRIC_COST = "Cost ($)"


def render_pipeline_slot(slot, name: str, status: str, result: PipelineResult | None = None, error: Exception | None = None) -> None:
    meta = PIPELINE_META[name]
    with slot.container():
        st.markdown(
            f'<div class="pipeline-card" style="--card-accent:{meta["color"]}"><h3>{meta["icon"]} {name}</h3><p>{meta["desc"]}</p></div>',
            unsafe_allow_html=True,
        )
        if status == "running":
            st.info("Running...")
        elif status == "error":
            st.error(f"Failed: {error}")
        elif status == "done" and result is not None:
            st.success(f"Done in {result.latency_s:.2f}s")
            answer = html.escape(result.answer).replace("\n", "<br>")
            st.markdown(
                f'<div class="answer-card" style="--card-accent:{meta["color"]}"><h4>Answer</h4><p>{answer}</p></div>',
                unsafe_allow_html=True,
            )
            if result.finish_reason and result.finish_reason != "stop":
                st.caption(f"⚠️ finish_reason: {result.finish_reason} (answer may be incomplete)")
        else:
            st.caption("Not available")


def render_result_cards(results: dict[str, PipelineResult], errors: dict[str, Exception]) -> None:
    st.divider()
    st.markdown('<h5 class="section-label">Pipeline results:</h5>', unsafe_allow_html=True)
    st.space()
    columns = st.columns(3)
    for name, column in zip(PIPELINES, columns):
        slot = column.empty()
        if name in errors:
            status = "error"
        elif name in results:
            status = "done"
        else:
            status = "idle"
        render_pipeline_slot(slot, name, status, results.get(name), errors.get(name))


def render_metrics(results: dict[str, PipelineResult]) -> None:
    metric_data = [
        {"Pipeline": name, "Prompt Tokens": results[name].prompt_tokens, "Completion Tokens": results[name].completion_tokens, "Total Tokens": results[name].total_tokens, METRIC_LATENCY: results[name].latency_s, METRIC_COST: results[name].cost}
        for name in PIPELINES if name in results
    ]
    if not metric_data:
        st.info("No successful pipeline runs to report metrics for.")
        return

    df = pd.DataFrame(metric_data).set_index("Pipeline")
    st.space()
    st.dataframe(df.style.format({METRIC_COST: "{:.6f}", METRIC_LATENCY: "{:.2f}"}), width="stretch")
    order = [name for name in PIPELINES if name in df.index]
    colors = [PIPELINE_META[name]["color"] for name in order]
    st.space()
    for column, metric in zip(st.columns(3), ("Total Tokens", METRIC_LATENCY, METRIC_COST)):
        with column:
            st.caption(metric)
            chart = alt.Chart(df.reset_index()).mark_bar().encode(
                x=alt.X("Pipeline:N", sort=order, title=None),
                y=alt.Y(f"{metric}:Q", title=None),
                color=alt.Color("Pipeline:N", scale=alt.Scale(domain=order, range=colors), legend=None),
                tooltip=["Pipeline", metric],
            ).properties(height=269)
            st.altair_chart(chart, width="stretch")


def render_context(results: dict[str, PipelineResult]) -> None:
    names = [name for name in ("Basic RAG", "GraphRAG") if name in results and results[name].retrieved_context]
    if not names:
        st.info("No retrieved context available for this run.")
        return
    for name in names:
        context = results[name].retrieved_context
        line_count = len(context.splitlines())
        st.space()
        st.markdown(f"##### {PIPELINE_META[name]['icon']} {name}")
        st.caption(f"{len(context):,} characters retrieved")
        st.code(context, language=None, line_numbers=True, height=500 if line_count > 10 else "content")
        st.download_button("⬇️ Download full context", data=context, file_name=f"{name.lower().replace(' ', '_')}_context.txt", mime="text/plain", key=f"download_{name}", width="content")
