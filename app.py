import asyncio
import html

import altair as alt
import pandas as pd
import streamlit as st

from llm.ag_llm import PipelineResult
import tg_llm.api as pipeline1
import tg_rag.api as pipeline2
import tg_graph_rag.api as pipeline3


PIPELINES = ("LLM-Only", "Basic RAG", "GraphRAG")
PIPELINE_META = {
    "LLM-Only": {"desc": "Parametric model knowledge", "color": "#8dc274", "icon": "🧠"},
    "Basic RAG": {"desc": "Native vector top-k retrieval", "color": "#5dd6e6", "icon": "🔎"},
    "GraphRAG": {"desc": "Vector seeds plus graph hops", "color": "#fda367", "icon": "🐯"},
}
EXAMPLE_QUESTIONS = [
    "What compounds treat epilepsy and what are their side effects?",
    "Which genes are associated with Parkinson's disease?",
    "What anatomical structures express the gene BRCA1?",
    "What gene id is linked with protein phosphatase, Mg2+/Mn2+ dependent, 1A?",
]

st.set_page_config(page_title="GraphRAG Benchmark", layout="wide", page_icon="🐯")
st.markdown(
    """
    <style>
    .block-container { max-width: 1440px; padding-top: 2rem; }
    .hero {
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1.4rem;
        padding: 0.4rem 0 1.25rem;
    }
    .hero-kicker {
        color: #2a9d8f; font-size: 0.76rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .hero h1 { font-size: 2.35rem; margin: 0.25rem 0 0.35rem; }
    .hero p { opacity: 0.75; font-size: 1rem; margin: 0; }
    .section-label {
        opacity: 0.7; font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase; margin: 1.4rem 0 0.55rem;
    }
    .pipeline-card {
        border: 1px solid rgba(128, 128, 128, 0.25); border-left: 3px solid var(--card-accent, #2a9d8f);
        border-radius: 10px; padding: 0.9rem 1rem; min-height: 5.5rem;
        transition: border-color 0.2s ease;
    }
    .pipeline-card h3 { font-size: 1rem; margin: 0 0 0.3rem; }
    .pipeline-card p { opacity: 0.75; font-size: 0.82rem; margin: 0; }
    .answer-card {
        border: 1px solid rgba(128, 128, 128, 0.25); border-top: 3px solid var(--card-accent, #e76f51);
        border-radius: 10px; padding: 1rem; min-height: 13rem;
    }
    .answer-card h4 { font-size: 0.9rem; margin: 0 0 0.65rem; }
    .answer-card p { line-height: 1.55; }
    .stButton > button { border-radius: 8px; font-weight: 700; }
    .chip-row .stButton > button {
        border-radius: 999px; font-weight: 500; font-size: 0.78rem;
        padding: 0.2rem 0.85rem;
    }
    </style>
    <div class="hero">
        <div class="hero-kicker">TigerGraph retrieval laboratory</div>
        <h1>🐯 GraphRAG Inference Benchmark</h1>
        <p>Compare parametric knowledge, native vector retrieval, and multi-hop graph evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration", icon="⚙️")
    st.divider()
    model = st.selectbox("LLM Model", ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"])
    top_k = st.slider("Top-K (retrieval)", 1, 20, 5)
    num_hops = st.slider("Graph Hops (GraphRAG)", 0, 5, 2)
    num_seen_min = st.slider("Min Seen (GraphRAG)", 1, 5, 1)

    st.caption("Tune retrieval depth and model behavior before running a comparison.")
    st.markdown("---")
    with st.expander("ℹ️ Instructions"):
        st.markdown(
            """
            ### Instructions
            1. Ensure your existing Savanna graph and schema are loaded.
            2. Set your environment variables:
                - `TG_API_KEY` – TigerGraph API token
                - `TG_HOST` – Savanna host URL
                - `TG_GRAPH` – Existing graph name
                - `LLM_API_KEY` – OpenAI secret key for LLM access
            3. Enter a question in the text area.
            4. Click "Run All Pipelines" to execute all three pipelines and compare results.

            """
        )

# ── Query input ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Example Questions</div>', unsafe_allow_html=True)


def _set_example_question(text: str) -> None:
    st.session_state["question_input"] = text


st.markdown('<div class="chip-row">', unsafe_allow_html=True)
chip_cols = st.columns(len(EXAMPLE_QUESTIONS) + 1)
for chip_col, example in zip(chip_cols, EXAMPLE_QUESTIONS):
    with chip_col:
        st.button(
            example if len(example) <= 42 else example[:39] + "…",
            key=f"example_{example}",
            on_click=_set_example_question,
            args=(example,),
            help=example,
        )
with chip_cols[-1]:
    if st.button("🗑️ Clear results", key="clear_results"):
        st.session_state.pop("results", None)
        st.session_state.pop("errors", None)
        st.session_state.pop("asked_question", None)
st.markdown("</div>", unsafe_allow_html=True)

question = st.text_area(
    "Biomedical question",
    height=100,
    placeholder="e.g. What compounds treat epilepsy and what are their side effects?",
    key="question_input",
)


def _render_pipeline_slot(
    slot: "st.delta_generator.DeltaGenerator",
    name: str,
    status: str,
    result: PipelineResult | None = None,
    error: Exception | None = None,
) -> None:
    """Render one pipeline's card + answer into its placeholder; called live as each finishes."""
    meta = PIPELINE_META[name]
    with slot.container():
        st.markdown(
            f'<div class="pipeline-card" style="--card-accent:{meta["color"]}">'
            f'<h3>{meta["icon"]} {name}</h3><p>{meta["desc"]}</p></div>',
            unsafe_allow_html=True,
        )
        if status == "running":
            st.info("Running...")
        elif status == "error":
            st.error(f"Failed: {error}")
        elif status == "done" and result is not None:
            st.success(f"Done in {result.latency_s:.2f}s")
            safe_answer = html.escape(result.answer).replace("\n", "<br>")
            st.markdown(
                f'<div class="answer-card" style="--card-accent:{meta["color"]}">'
                f'<h4>Answer</h4><p>{safe_answer}</p></div>',
                unsafe_allow_html=True,
            )
            if result.finish_reason and result.finish_reason != "stop":
                st.caption(f"⚠️ finish_reason: {result.finish_reason} (answer may be incomplete)")
        else:
            st.caption("Not available")


async def _run_all_pipelines(
    question: str,
    model: str,
    top_k: int,
    num_hops: int,
    num_seen_min: int,
    slots: dict[str, object],
) -> tuple[dict[str, PipelineResult], dict[str, Exception]]:
    async def run_one(name: str, task):
        try:
            return name, await task
        except Exception as error:
            return name, error

    tasks = [
        asyncio.create_task(
            run_one("LLM-Only", pipeline1.async_query(question, model=model))
        ),
        asyncio.create_task(
            run_one(
                "Basic RAG",
                pipeline2.async_query(question, top_k=top_k, model=model),
            )
        ),
        asyncio.create_task(
            run_one(
                "GraphRAG",
                pipeline3.async_query(
                    question,
                    top_k=top_k,
                    num_hops=num_hops,
                    num_seen_min=num_seen_min,
                    model=model,
                ),
            )
        ),
    ]
    results: dict[str, PipelineResult] = {}
    errors: dict[str, Exception] = {}
    # update each pipeline's card the moment it finishes, instead of waiting for all three
    for completed in asyncio.as_completed(tasks):
        name, outcome = await completed
        if isinstance(outcome, Exception):
            errors[name] = outcome
            _render_pipeline_slot(slots[name], name, "error", error=outcome)
        else:
            results[name] = outcome
            _render_pipeline_slot(slots[name], name, "done", result=outcome)
    return results, errors


run_clicked = st.button("🚀 Run All Pipelines", type="primary", use_container_width=True)

results: dict[str, PipelineResult] = st.session_state.get("results", {})
errors: dict[str, Exception] = st.session_state.get("errors", {})
asked_question: str = st.session_state.get("asked_question", "")

if run_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.markdown('<div class="section-label">Pipeline results</div>', unsafe_allow_html=True)
    status_cols = st.columns(3)
    slots = {name: col.empty() for name, col in zip(PIPELINES, status_cols)}
    for name in PIPELINES:
        _render_pipeline_slot(slots[name], name, "running")

    results, errors = asyncio.run(
        _run_all_pipelines(question, model, top_k, num_hops, num_seen_min, slots)
    )

    st.session_state["results"] = results
    st.session_state["errors"] = errors
    st.session_state["asked_question"] = question
    asked_question = question
elif results or errors:
    st.markdown('<div class="section-label">Pipeline results</div>', unsafe_allow_html=True)
    status_cols = st.columns(3)
    for name, col in zip(PIPELINES, status_cols):
        slot = col.empty()
        if name in errors:
            _render_pipeline_slot(slot, name, "error", error=errors[name])
        elif name in results:
            _render_pipeline_slot(slot, name, "done", result=results[name])
        else:
            _render_pipeline_slot(slot, name, "idle")

if results or errors:
    st.markdown("---")
    # st.caption(f"Showing results for: *“{asked_question}”*")

    tab_metrics, tab_context = st.tabs(["📊 Metrics", "🔍 Retrieved Context"])

    # ── Metrics tab ──────────────────────────────────────────────────────
    with tab_metrics:
        metric_data = []
        for name in PIPELINES:
            if name in results:
                r = results[name]
                metric_data.append({
                    "Pipeline": name,
                    "Prompt Tokens": r.prompt_tokens,
                    "Completion Tokens": r.completion_tokens,
                    "Total Tokens": r.total_tokens,
                    "Latency (s)": r.latency_s,
                    "Cost ($)": r.cost,
                })

        if metric_data:
            df = pd.DataFrame(metric_data).set_index("Pipeline")
            st.dataframe(
                df.style.format({"Cost ($)": "{:.6f}", "Latency (s)": "{:.2f}"}),
                use_container_width=True,
            )

            pipeline_order = [name for name in PIPELINES if name in df.index]
            pipeline_colors = [PIPELINE_META[name]["color"] for name in pipeline_order]

            def _ordered_bar_chart(column: str) -> alt.Chart:
                return (
                    alt.Chart(df.reset_index())
                    .mark_bar()
                    .encode(
                        x=alt.X("Pipeline:N", sort=pipeline_order, title=None),
                        y=alt.Y(f"{column}:Q", title=None),
                        color=alt.Color(
                            "Pipeline:N",
                            scale=alt.Scale(domain=pipeline_order, range=pipeline_colors),
                            legend=None,
                        ),
                        tooltip=["Pipeline", column],
                    )
                    .properties(height=220)
                )

            chart_cols = st.columns(3)
            with chart_cols[0]:
                st.caption("Total tokens")
                st.altair_chart(_ordered_bar_chart("Total Tokens"), use_container_width=True)
            with chart_cols[1]:
                st.caption("Latency (s)")
                st.altair_chart(_ordered_bar_chart("Latency (s)"), use_container_width=True)
            with chart_cols[2]:
                st.caption("Cost ($)")
                st.altair_chart(_ordered_bar_chart("Cost ($)"), use_container_width=True)
        else:
            st.info("No successful pipeline runs to report metrics for.")

    # ── Retrieved context tab ──────────────────────────────────────────
    with tab_context:
        context_pipelines = [
            name for name in ("Basic RAG", "GraphRAG")
            if name in results and results[name].retrieved_context
        ]
        if not context_pipelines:
            st.info("No retrieved context available for this run.")
        else:
            ctx_cols = st.columns(len(context_pipelines))
            for name, col in zip(context_pipelines, ctx_cols):
                meta = PIPELINE_META[name]
                ctx = results[name].retrieved_context
                with col:
                    st.markdown(f"##### {meta['icon']} {name}")
                    st.caption(f"{len(ctx):,} characters retrieved")
                    st.code(ctx, language=None, wrap_lines=True, height=500)
                    st.download_button(
                        "⬇️ Download full context",
                        data=ctx,
                        file_name=f"{name.lower().replace(' ', '_')}_context.txt",
                        mime="text/plain",
                        key=f"download_{name}",
                        use_container_width=True,
                    )
else:
    st.info("Enter a question above and click **Run All Pipelines** to see results here.")
