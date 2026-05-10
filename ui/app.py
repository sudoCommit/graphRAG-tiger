import streamlit as st

from llm.ag_llm import PipelineResult
import tg_llm.api as pipeline1
import tg_rag.api as pipeline2
import tg_graph_rag.api as pipeline3

st.set_page_config(page_title="GraphRAG Benchmark", layout="wide")
st.title("🐯 GraphRAG Inference Benchmark")
st.markdown("Compare **LLM-Only** vs **Basic RAG** vs **GraphRAG** side-by-side")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    model = st.selectbox("LLM Model", ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"])
    top_k = st.slider("Top-K (retrieval)", 1, 20, 5)
    num_hops = st.slider("Graph Hops (GraphRAG)", 0, 5, 2)
    num_seen_min = st.slider("Min Seen (GraphRAG)", 1, 5, 2)

# ── Query input ──────────────────────────────────────────────────────────────
question = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="e.g. What compounds treat epilepsy and what are their side effects?",
)

if st.button("🚀 Run All Pipelines", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    results: dict[str, PipelineResult] = {}

    # ── Run pipelines ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Pipeline 1: LLM-Only")
        with st.spinner("Running..."):
            try:
                results["LLM-Only"] = pipeline1.query(question, model=model)
                st.success(f"Done in {results['LLM-Only'].latency_s}s")
            except Exception as e:
                st.error(f"Failed: {e}")

    with col2:
        st.subheader("Pipeline 2: Basic RAG")
        with st.spinner("Running..."):
            try:
                results["Basic RAG"] = pipeline2.query(
                    question, top_k=top_k, model=model,
                )
                st.success(f"Done in {results['Basic RAG'].latency_s}s")
            except Exception as e:
                st.error(f"Failed: {e}")

    with col3:
        st.subheader("Pipeline 3: GraphRAG")
        with st.spinner("Running..."):
            try:
                results["GraphRAG"] = pipeline3.query(
                    question,
                    top_k=top_k,
                    num_hops=num_hops,
                    num_seen_min=num_seen_min,
                    model=model,
                )
                st.success(f"Done in {results['GraphRAG'].latency_s}s")
            except Exception as e:
                st.error(f"Failed: {e}")

    # ── Answers ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Answers")
    ans_cols = st.columns(3)
    for name, col in zip(["LLM-Only", "Basic RAG", "GraphRAG"], ans_cols):
        with col:
            st.markdown(f"**{name}**")
            if name in results:
                st.write(results[name].answer)
            else:
                st.warning("Not available")

    # ── Metrics table ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Metrics Comparison")

    metric_data = []
    for name in ["LLM-Only", "Basic RAG", "GraphRAG"]:
        if name in results:
            r = results[name]
            metric_data.append({
                "Pipeline": name,
                "Prompt Tokens": r.prompt_tokens,
                "Completion Tokens": r.completion_tokens,
                "Total Tokens": r.total_tokens,
                "Latency (s)": r.latency_s,
                "Cost ($)": f"{r.cost:.6f}",
            })
    if metric_data:
        st.table(metric_data)

    # ── Usage details from response payload ─────────────────────────────
    for name in ["LLM-Only", "Basic RAG", "GraphRAG"]:
        if name in results:
            r = results[name]
            usage_payload = {
                "id": r.response_id,
                "model": r.model,
                "created": r.created,
                "service_tier": r.service_tier,
                "system_fingerprint": r.system_fingerprint,
                "finish_reason": r.finish_reason,
                "prompt_tokens_details": r.prompt_tokens_details,
                "completion_tokens_details": r.completion_tokens_details,
                "latency_checkpoint": r.latency_checkpoint,
            }
            has_details = any([
                usage_payload["id"],
                usage_payload["finish_reason"],
                usage_payload["prompt_tokens_details"],
                usage_payload["completion_tokens_details"],
                usage_payload["latency_checkpoint"],
            ])
            if has_details:
                with st.expander(f"🧾 {name} — Usage Details"):
                    st.json(usage_payload)

    # ── Token reduction highlight ────────────────────────────────────────
    if "Basic RAG" in results and "GraphRAG" in results:
        rag = results["Basic RAG"].total_tokens
        graph = results["GraphRAG"].total_tokens
        if rag > 0:
            reduction = ((rag - graph) / rag) * 100
            st.metric(
                "📊 Token Reduction (GraphRAG vs Basic RAG)",
                f"{reduction:.1f}%",
                delta=f"{rag - graph} tokens saved",
                delta_color="normal" if reduction > 0 else "inverse",
            )

    # ── Retrieved context (expandable) ───────────────────────────────────
    for name in ["Basic RAG", "GraphRAG"]:
        if name in results and results[name].retrieved_context:
            with st.expander(f"🔍 {name} — Retrieved Context"):
                st.text(results[name].retrieved_context[:3000])
