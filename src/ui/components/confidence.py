"""Confidence dashboard with gauge and probability bars."""

from __future__ import annotations

import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.ui.utils.disclaimer import render_disclaimer


def render_confidence_dashboard(fusion) -> None:
    if fusion is None:
        st.warning("Fusion prediction unavailable. Run full analysis or verify model checkpoint.")
        return

    render_disclaimer()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Predicted Risk", fusion.predicted_label)
    with col2:
        st.metric("Overall Confidence", f"{fusion.confidence:.1%}")
    with col3:
        margin = _probability_margin(fusion.probabilities)
        st.metric("Prediction Certainty", f"{margin:.1%}")

    if HAS_PLOTLY:
        _gauge_chart(fusion.confidence)
        _probability_bars(fusion.probabilities)
    else:
        for label, prob in fusion.probabilities.items():
            st.progress(float(prob), text=f"{label}: {prob:.1%}")


def _gauge_chart(confidence: float) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": "Model Confidence", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#4361ee"},
            "steps": [
                {"range": [0, 40], "color": "#fef3c7"},
                {"range": [40, 70], "color": "#bfdbfe"},
                {"range": [70, 100], "color": "#bbf7d0"},
            ],
            "threshold": {"line": {"color": "#ef476f", "width": 4}, "value": 85},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width='stretch')


def _probability_bars(probabilities: dict[str, float]) -> None:
    import plotly.graph_objects as go

    labels = list(probabilities.keys())
    values = [probabilities[k] for k in labels]
    colors = ["#2a9d8f", "#e9c46a", "#e63946"]
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors[: len(labels)],
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Risk Class Probabilities",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=260,
        margin=dict(l=20, r=40, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width='stretch')


def _probability_margin(probabilities: dict[str, float]) -> float:
    vals = sorted(probabilities.values(), reverse=True)
    if len(vals) < 2:
        return vals[0] if vals else 0.0
    return vals[0] - vals[1]
