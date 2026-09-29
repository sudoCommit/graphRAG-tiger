import asyncio

import streamlit as st

from pre_defined.evaluation import load_questions, score_answer
from pre_defined.main import _connection, query_graph, query_graph_response, retrieval_diagnostics
from pre_defined.structured_query import try_structured_answer
from ui.config import (
    DEFAULT_EDGE_LIMIT,
    DEFAULT_NUM_HOPS,
    DEFAULT_NUM_SEEN_MIN,
    DEFAULT_TOP_K,
    EXAMPLE_QUESTIONS,
    MODEL_OPTIONS,
    PIPELINES,
)
from ui.pipelines import run_all
from ui.styles import render_page_header
from ui.views import render_context, render_metrics, render_pipeline_slot, render_result_cards


SIDEBAR_WIDGET_KEYS = ("model_select", "top_k", "num_hops", "num_seen_min", "edge_limit")
PREDEFINED_SECTION = "Pre-defined corpus"
CORRECT_METRIC = "Correct"
# top_k=5 can't cover an entire event category; widen retrieval for count/max questions.
BROAD_COVERAGE_QTYPES = {"aggregation", "superlative"}
BROAD_COVERAGE_TOP_K = 20


def _render_predefined_query_tab(method: str, top_k: int, num_hops: int, num_seen_min: int) -> None:
    question = st.text_area(
        "Question",
        height=120,
        placeholder="e.g. Who won the gold medal in the men's 20 kilometres walk at the 2012 Olympics?",
        key="predefined_question",
    )
    if st.button("🚀 Ask predefined corpus", type="primary", key="predefined_run"):
        if not question.strip():
            st.warning("Please enter a question.")
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


def _score_structured_item(item: dict, answer: str) -> dict[str, object]:
    result = score_answer(answer, item.get("answer", []), item["qtype"])
    return {
        "QID": item["qid"],
        "Type": item["qtype"],
        "Status": "ok",
        CORRECT_METRIC: result["correct"],
        "Method": f"structured+{result['method']}",
        "Confidence": result["confidence"],
        "Answer": answer,
    }


def _score_evaluation_item(
    connection, item: dict, method: str, top_k: int, num_hops: int, num_seen_min: int
) -> dict[str, object]:
    # aggregation/superlative questions reference facts no top_k retrieval can
    # fully cover; answer them deterministically from parsed infobox fields
    # instead of asking the LLM to guess over a handful of chunks.
    structured_answer = try_structured_answer(item["question"], item["qtype"])
    if structured_answer is not None:
        return _score_structured_item(item, structured_answer)

    effective_top_k = max(top_k, BROAD_COVERAGE_TOP_K) if item["qtype"] in BROAD_COVERAGE_QTYPES else top_k
    try:
        response = query_graph_response(
            connection, item["question"], method, effective_top_k, num_hops, num_seen_min
        )
        answer = response["response"]
        result = score_answer(answer, item.get("answer", []), item["qtype"])
        return {
            "QID": item["qid"],
            "Type": item["qtype"],
            "Status": "ok",
            CORRECT_METRIC: result["correct"],
            "Method": result["method"],
            "Confidence": result["confidence"],
            "Answer": answer,
            **retrieval_diagnostics(response, item.get("gold_doc_ids")),
        }
    except Exception as error:
        return {
            "QID": item["qid"],
            "Type": item["qtype"],
            "Status": f"error: {error}",
            CORRECT_METRIC: None,
            "Method": "n/a",
            "Confidence": None,
            "Answer": "",
        }


def _render_evaluation_tab(method: str, top_k: int, num_hops: int, num_seen_min: int) -> None:
    dataset = st.radio(
        "Question set",
        ("eval_public", "eval_hidden"),
        format_func=lambda value: "Public (scored)" if value == "eval_public" else "Hidden (execution only)",
        horizontal=True,
        key="predefined_eval_dataset",
    )
    questions = load_questions(dataset)
    st.caption(f"{len(questions)} evaluation questions loaded.")
    if st.button("▶️ Run evaluation", type="primary", key="predefined_eval_run"):
        connection = _connection()
        rows: list[dict[str, object]] = []
        progress = st.progress(0, text="Starting evaluation...")
        for index, item in enumerate(questions, start=1):
            rows.append(_score_evaluation_item(connection, item, method, top_k, num_hops, num_seen_min))
            progress.progress(index / len(questions), text=f"Completed {index} of {len(questions)}")
        st.session_state["predefined_eval_rows"] = rows

    rows = st.session_state.get("predefined_eval_rows")
    if not rows:
        return
    successful = [row for row in rows if row["Status"] == "ok"]
    scored = [row for row in successful if row[CORRECT_METRIC] is not None]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Executed", f"{len(successful)}/{len(rows)}")
    if scored:
        metric_columns[1].metric(CORRECT_METRIC, f"{sum(row[CORRECT_METRIC] for row in scored) / len(scored):.1%}")
        metric_columns[2].metric("Avg confidence", f"{sum(row['Confidence'] for row in scored) / len(scored):.1%}")
    else:
        metric_columns[1].metric(CORRECT_METRIC, "N/A")
        metric_columns[2].metric("Avg confidence", "N/A")
    metric_columns[3].metric("Errors", len(rows) - len(successful))
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_predefined_section() -> None:
    st.markdown(f'<div class="section-label">{PREDEFINED_SECTION}</div>', unsafe_allow_html=True)
    st.subheader("Wikipedia GraphRAG")
    st.caption("Ask questions against the pre-loaded Wikipedia corpus using TigerGraph GraphRAG.")

    with st.sidebar:
        st.header(PREDEFINED_SECTION, icon="📚")
        method = st.selectbox(
            "Retrieval method",
            ("hybrid", "similarity", "community"),
            format_func=str.title,
            key="predefined_method",
        )
        top_k = st.slider("Top-K", 1, 20, 5, key="predefined_top_k")
        num_hops = st.slider("Graph hops", 0, 10, 2, key="predefined_num_hops")
        num_seen_min = st.slider("Minimum paths", 1, 5, 1, key="predefined_num_seen_min")

    query_tab, evaluation_tab = st.tabs(["💬 Normal query", "📈 Evaluation"])
    with query_tab:
        _render_predefined_query_tab(method, top_k, num_hops, num_seen_min)
    with evaluation_tab:
        _render_evaluation_tab(method, top_k, num_hops, num_seen_min)


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


def main() -> None:
    st.set_page_config(page_title="GraphRAG Benchmark", layout="wide", page_icon="🐯")
    render_page_header()
    with st.sidebar:
        section = st.radio(
            "Section",
            ("Pipeline comparison", PREDEFINED_SECTION),
            key="active_section",
        )
        st.divider()
    if section == PREDEFINED_SECTION:
        _render_predefined_section()
        return

    model, top_k, num_hops, num_seen_min, edge_limit = _render_sidebar()
    question = _render_question_input()
    run_clicked = st.button("🚀 Run All Pipelines", type="primary", width="stretch")
    results = st.session_state.get("results", {})
    errors = st.session_state.get("errors", {})

    if run_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
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
