"""Shared Streamlit session state for the analysis workstation."""

from __future__ import annotations

from typing import Any

import streamlit as st

SESSION_KEY = "cortex_analysis"


def init_session() -> None:
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = {
            "patient_id": None,
            "mri": None,
            "mask": None,
            "report_text": None,
            "stats": None,
            "bboxes": None,
            "lesions": None,
            "image_features": None,
            "text_features": None,
            "clinical_row": None,
            "clinical_entities": None,
            "fusion": None,
            "unified_repr": None,
            "gradcam": None,
            "gradcam_target": "ET",
            "shap_importance": None,
            "ai_summary": None,
            "similar_patients": None,
            "pipeline_log": [],
            "analysis_complete": False,
        }


def get_analysis() -> dict[str, Any]:
    init_session()
    return st.session_state[SESSION_KEY]


def update_analysis(**kwargs: Any) -> None:
    state = get_analysis()
    state.update(kwargs)


def reset_analysis() -> None:
    st.session_state[SESSION_KEY] = {}
    init_session()


def is_analysis_ready() -> bool:
    state = get_analysis()
    return bool(state.get("analysis_complete") and state.get("mask") is not None)
