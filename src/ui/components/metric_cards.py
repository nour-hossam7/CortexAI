"""Dashboard metric cards and pipeline diagram."""

from __future__ import annotations

import streamlit as st

from src.ui.utils.theme import get_theme


def metric_card(label: str, value: str, icon: str = "📊") -> None:
    t = get_theme()
    st.markdown(
        f"""
        <div class="cortex-card">
            <div class="cortex-metric-label">{icon} {label}</div>
            <div class="cortex-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_flow() -> None:
    steps = [
        "MRI",
        "Segmentation",
        "Feature Extraction",
        "Clinical Report",
        "ClinicalBERT",
        "Fusion",
        "Clinical Decision",
        "Explainability",
        "Medical Report",
    ]
    t = get_theme()
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        with col:
            st.markdown(
                f"<div style='text-align:center;font-size:0.7rem;color:{t['muted']};'>"
                f"<div style='color:{t['accent']};font-weight:600;'>{step}</div>↓</div>",
                unsafe_allow_html=True,
            )
