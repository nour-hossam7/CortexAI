"""Settings — theme, device, model paths."""

from __future__ import annotations

import streamlit as st

from src.ui.bootstrap import setup_page
from src.ui.utils.models_cache import get_device_name, load_thresholds
from src.ui.utils.paths import get_paths
from src.ui.utils.session import reset_analysis

setup_page("Settings", icon="⚙️", wide=False)

st.title("Settings")

st.subheader("Compute Device")
st.info(get_device_name())

st.subheader("Model Paths")
paths = get_paths()
st.code(
    f"SegResNet: {paths.segresnet_checkpoint}\n"
    f"Fusion:    {paths.fusion_checkpoint}\n"
    f"Scaler:    {paths.clinical_scaler}\n"
    f"BioBERT:   {paths.nlp_embeddings_dir}",
    language="text",
)

st.subheader("Severity Thresholds")
thr = load_thresholds()
if thr:
    st.json(thr)
else:
    st.warning("severity_thresholds.json not found.")

st.subheader("Session")
if st.button("Reset analysis session", type="secondary"):
    reset_analysis()
    st.success("Session cleared.")
    st.rerun()
