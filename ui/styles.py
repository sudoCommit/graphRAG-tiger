import streamlit as st


STYLE = """
<style>
.block-container { max-width: 1440px; padding-top: 2rem; }
.hero { border-bottom: 1px solid rgba(128, 128, 128, 0.25); margin-bottom: 1.4rem; padding: 0.4rem 0 1.25rem; }
.hero-kicker { color: #2a9d8f; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
.hero h1 { font-size: 2.35rem; margin: 0.25rem 0 0.35rem; }
.hero p { opacity: 0.75; font-size: 1rem; margin: 0; }
.section-label { opacity: 0.7; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin: 1.4rem 0 0.55rem; }
.pipeline-card { border: 1px solid rgba(128, 128, 128, 0.25); border-left: 3px solid var(--card-accent, #2a9d8f); border-radius: 10px; padding: 0.9rem 1rem; min-height: 5.5rem; }
.pipeline-card h3 { font-size: 1rem; margin: 0 0 0.3rem; }
.pipeline-card p { opacity: 0.75; font-size: 0.82rem; margin: 0; }
.answer-card { border: 1px solid rgba(128, 128, 128, 0.25); border-top: 3px solid var(--card-accent, #e76f51); border-radius: 10px; padding: 1rem; min-height: 13rem; }
.answer-card h4 { font-size: 0.9rem; margin: 0 0 0.65rem; }
.answer-card p { line-height: 1.55; }
.stButton > button { border-radius: 8px; font-weight: 700; }
.chip-row .stButton > button { border-radius: 999px; font-weight: 500; font-size: 0.78rem; padding: 0.2rem 0.85rem; }
</style>
"""


def render_page_header() -> None:
    st.markdown(
        STYLE
        + '<div class="hero"><div class="hero-kicker">TigerGraph retrieval laboratory</div>'
        '<h1>🐯 GraphRAG Inference Benchmark</h1>'
        '<p>Compare parametric knowledge, native vector retrieval, and multi-hop graph evidence.</p></div>',
        unsafe_allow_html=True,
    )
