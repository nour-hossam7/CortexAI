"""Fusion AI — multimodal risk prediction."""

from __future__ import annotations

import streamlit as st

from src.ui.bootstrap import setup_page
from src.ui.components.confidence import render_confidence_dashboard
from src.ui.utils.disclaimer import render_disclaimer
from src.ui.utils.session import get_analysis

setup_page("Fusion AI", icon="🔗")

st.title("Fusion AI — Clinical Decision Support")
render_disclaimer()

analysis = get_analysis()

if not analysis.get("analysis_complete"):
    st.warning("Complete multimodal analysis from the MRI Workstation page first.")
    st.stop()

fusion = analysis.get("fusion")
clinical = analysis.get("clinical_row") or {}

c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Multimodal Inputs")
    st.markdown("**Image features:** SegResNet bottleneck (256-d)")
    st.markdown("**Text features:** BioBERT mean-pooled embedding (768-d)")
    st.markdown("**Clinical vector:** scaled rule-based + NLP features")

    if clinical:
        with st.expander("Clinical feature values (unscaled)"):
            st.json({k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in clinical.items()})

with c2:
    st.subheader("Prediction")
    if fusion:
        st.markdown(f"### {fusion.predicted_label}")
        st.progress(fusion.confidence, text=f"Confidence: {fusion.confidence:.1%}")
        for label, prob in fusion.probabilities.items():
            st.markdown(f"- **{label}:** {prob:.1%}")
    else:
        st.error("Fusion model unavailable. Verify `models/fusion/best_decision_model.pth`.")

st.markdown("---")
render_confidence_dashboard(fusion)

st.markdown("---")
st.subheader("Clinical Decision")
if fusion:
    decision_map = {
        "Low Risk": "Continued surveillance with standard interval MRI per institutional protocol.",
        "Medium Risk": "Consider short-interval imaging and multidisciplinary review.",
        "High Risk": "Expedited neuro-oncology evaluation and treatment planning recommended.",
    }
    st.info(decision_map.get(fusion.predicted_label, "Clinical correlation required."))

    st.subheader("Prediction Explanation")
    st.markdown(
        "The fusion encoder projected imaging and text modalities into a shared 256-dimensional "
        "representation, then the decision head combined this with 13 clinical features to produce "
        f"a **{fusion.predicted_label}** classification with **{fusion.confidence:.1%}** softmax confidence."
    )
