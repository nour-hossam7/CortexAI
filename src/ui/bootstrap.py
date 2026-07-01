"""Shared page bootstrap: theme, session, sidebar."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.ui.utils.disclaimer import MEDICAL_DISCLAIMER
from src.ui.utils.models_cache import get_device_name, get_paths
from src.ui.utils.paths import load_dataset_info
from src.ui.utils.session import get_analysis, init_session
from src.ui.utils.theme import apply_theme


def setup_page(title: str, icon: str = "🧠", wide: bool = True) -> None:
    layout = "wide" if wide else "centered"
    st.set_page_config(page_title=f"CortexAI | {title}", page_icon=icon, layout=layout, initial_sidebar_state="expanded")
    init_session()
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "dark"
    apply_theme()
    _render_sidebar()


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## CortexAI")
        st.caption("Multimodal Brain Tumor CDS")
        st.markdown("---")

        analysis = get_analysis()
        if analysis.get("analysis_complete"):
            st.success(f"Patient: {analysis.get('patient_id', '—')}")
        else:
            st.info("No active analysis")

        st.markdown("---")
        st.caption(MEDICAL_DISCLAIMER)

        info = load_dataset_info()
        pts = info.get("patients", {})
        st.caption(f"Cohort: {pts.get('total', 369)} patients")
