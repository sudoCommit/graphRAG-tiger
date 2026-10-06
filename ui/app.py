"""Streamlit router that switches between the pipeline comparison and pre-defined sections."""

import streamlit as st

from ui.comparison_view import render_comparison_section
from ui.config import COMPARISON_SECTION, PREDEFINED_SECTION
from ui.predefined_view import render_predefined_section
from ui.styles import render_page_header


def main() -> None:
    st.set_page_config(page_title="GraphRAG Benchmark", layout="wide", page_icon="🐯")
    render_page_header()
    with st.sidebar:
        section = st.radio(
            "Section",
            (COMPARISON_SECTION, PREDEFINED_SECTION),
            key="active_section",
        )
        st.divider()
    if section == PREDEFINED_SECTION:
        render_predefined_section()
    else:
        render_comparison_section()
