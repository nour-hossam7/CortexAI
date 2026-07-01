"""Explainability — Grad-CAM, SHAP, similar patients."""

from __future__ import annotations

import streamlit as st

from src.ui.bootstrap import setup_page
from src.ui.components.confidence import render_confidence_dashboard
from src.ui.components.mri_viewer import render_gradcam_panel
from src.ui.utils.disclaimer import render_disclaimer
from src.ui.utils.paths import get_paths
from src.ui.utils.session import get_analysis

setup_page("Explainability", icon="🔍")

st.title("Explainability & Interpretability")
render_disclaimer()

analysis = get_analysis()
paths = get_paths()

tab_gc, tab_shap, tab_sim, tab_fig = st.tabs(["Grad-CAM", "SHAP", "Similar Patients", "Saved Figures"])

with tab_gc:
    render_gradcam_panel(analysis)

with tab_shap:
    st.subheader("SHAP Clinical Feature Importance")
    shap_df = analysis.get("shap_importance")
    csv_path = paths.fusion_results / "shap_clinical_importance.csv"
    fig_path = paths.fusion_figures / "shap_clinical_importance.png"

    if shap_df is not None:
        st.bar_chart(shap_df.set_index("feature")["importance"])
        st.dataframe(shap_df, width='stretch', hide_index=True)
    elif csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        st.bar_chart(df.set_index("feature")["importance"])
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("Pre-computed SHAP outputs not found. Run fusion notebook 05 or disable this panel.")

    if fig_path.exists():
        st.image(str(fig_path), caption="SHAP Summary (saved)", width='stretch')

    for name in ["shap_beeswarm.png", "shap_waterfall_examples.png"]:
        p = paths.fusion_figures / name
        if p.exists():
            st.image(str(p), caption=name.replace("_", " ").title(), width='stretch')

with tab_sim:
    st.subheader("Similar Patient Retrieval")
    similar = analysis.get("similar_patients")
    if similar is not None and len(similar):
        st.dataframe(similar, width='stretch', hide_index=True)
        selected = st.selectbox("Open similar case", similar["patient_id"].tolist())
        if selected:
            st.markdown(f"**Selected:** `{selected}` — similarity {similar[similar['patient_id']==selected]['similarity'].iloc[0]}")
            st.caption("Full case viewer requires cohort MRI/report assets for the selected patient.")
    else:
        st.info("Similar patients appear after analysis when cohort representations or image features are available.")

with tab_fig:
    st.subheader("Evaluation Figures")
    fusion_figs = paths.list_figures(paths.fusion_figures)
    cv_figs = paths.list_figures(paths.cv_figures)
    if fusion_figs:
        for fig in fusion_figs:
            st.image(str(fig), caption=fig.stem, width='stretch')
    elif cv_figs:
        for fig in cv_figs:
            st.image(str(fig), caption=fig.stem, width='stretch')
    else:
        st.info("No saved explainability figures in reports/figures/.")

if analysis.get("fusion"):
    st.markdown("---")
    render_confidence_dashboard(analysis["fusion"])
