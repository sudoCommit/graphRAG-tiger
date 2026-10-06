"""Hetionet pipeline comparison section: LLM-Only, Basic RAG, and GraphRAG on one question."""

import asyncio

import streamlit as st

from ui.config import (
    DEFAULT_EDGE_LIMIT,
    DEFAULT_NUM_HOPS,
    DEFAULT_NUM_SEEN_MIN,
    DEFAULT_TOP_K,
    EMPTY_QUESTION_WARNING,
    EXAMPLE_QUESTIONS,
    MODEL_OPTIONS,
    PIPELINES,
)
from ui.pipelines import run_all
from ui.views import render_context, render_metrics, render_pipeline_slot, render_result_cards


def _render_sidebar() -> tuple[str, int, int, int, int]:
    with st.sidebar:
        st.header("Configuration", icon="⚙️")
        st.caption("Tune the parameters before running a comparison.")

        model = st.selectbox(
            "Chat model",
            MODEL_OPTIONS,
            help="Model used to generate all three answers.",
            key="model_select",
        )

        st.divider()
        with st.expander("Retrieval", expanded=True, icon="🔎"):
            top_k = st.slider(
                "Top-K seed vertices", 1, 20, DEFAULT_TOP_K,
                help="Number of nearest vector-search results used as seeds.",
                key="top_k",
            )

        with st.expander("GraphRAG traversal", expanded=True, icon="🕸️"):
            num_hops = st.slider(
                "Graph hops", 0, 10, DEFAULT_NUM_HOPS,
                help="Number of graph levels expanded from the vector seeds.",
                key="num_hops",
            )
            num_seen_min = st.slider(
                "Minimum paths to accept a vertex", 1, 5, DEFAULT_NUM_SEEN_MIN,
                help="A neighbor must be seen this many times before it is added to context.",
                key="num_seen_min",
            )
            edge_limit = st.slider(
                "Edges per vertex", 1, 30, DEFAULT_EDGE_LIMIT,
                help="Maximum outgoing edges inspected from each vertex at every hop.",
                key="edge_limit",
            )

        st.caption(
            f"Active · Top-K **{top_k}** · Hops **{num_hops}** · "
            f"Min seen **{num_seen_min}** · Edges/vertex **{edge_limit}**"
        )

        st.divider()
        with st.expander("Instructions", icon="ℹ️"):
            st.markdown(
                """
                1. Ensure your environment is set up correctly:
                    - Savanna graph and schema loaded
                    - Document embeddings loaded
                    - Vector indexes ready
                    - Native vector queries installed
                2. Enter a question in the text area.
                3. Click "Run All Pipelines" to execute all three pipelines and compare results.
                """,
            )
    return model, top_k, num_hops, num_seen_min, edge_limit


def _render_question_input() -> str:
    st.markdown('<div class="section-label">Example Questions</div>', unsafe_allow_html=True)

    def set_example(text: str) -> None:
        st.session_state["question_input"] = text

    columns = st.columns(len(EXAMPLE_QUESTIONS) + 1)
    for column, example in zip(columns, EXAMPLE_QUESTIONS):
        with column:
            st.button(example if len(example) <= 42 else example[:39] + "…", key=f"example_{example}", on_click=set_example, args=(example,), help=example)
    with columns[-1]:
        if st.button("🗑️ Clear results", key="clear_results"):
            for key in ("results", "errors", "asked_question"):
                st.session_state.pop(key, None)
    return st.text_area("Biomedical question", height=100, placeholder="e.g. What compounds treat epilepsy and what are their side effects?", key="question_input")


def render_comparison_section() -> None:
    model, top_k, num_hops, num_seen_min, edge_limit = _render_sidebar()
    question = _render_question_input()
    run_clicked = st.button("🚀 Run All Pipelines", type="primary", width="stretch")
    results = st.session_state.get("results", {})
    errors = st.session_state.get("errors", {})

    if run_clicked:
        if not question.strip():
            st.warning(EMPTY_QUESTION_WARNING)
            st.stop()
        slots = {name: column.empty() for name, column in zip(PIPELINES, st.columns(3))}
        for name in PIPELINES:
            render_pipeline_slot(slots[name], name, "running")
        results, errors = asyncio.run(run_all(question, model, top_k, num_hops, num_seen_min, edge_limit, slots, render_pipeline_slot))
        st.session_state["results"] = results
        st.session_state["errors"] = errors
    elif results or errors:
        render_result_cards(results, errors)

    if results or errors:
        st.space()
        metrics_tab, context_tab = st.tabs(["📊 Metrics", "🔍 Retrieved Context"])
        with metrics_tab:
            render_metrics(results)
        with context_tab:
            render_context(results)
    else:
        st.info("Enter a question above and click **Run All Pipelines** to see results here.")
