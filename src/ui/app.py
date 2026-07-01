"""CortexAI — Home Dashboard (use root app.py for streamlit run)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.ui.bootstrap import setup_page
from src.ui.components.metric_cards import metric_card, pipeline_flow
from src.ui.utils.disclaimer import render_disclaimer
from src.ui.utils.models_cache import get_device_name, get_paths
from src.ui.utils.paths import load_dataset_info
from src.ui.utils.session import get_analysis

setup_page("Dashboard", icon="🏠")

st.markdown(
    """
    <div class="cortex-hero">
        <h1>CortexAI</h1>
        <p style="font-size:1.15rem;opacity:0.85;margin-top:0.5rem;">
        Multimodal Brain Tumor Clinical Decision Support System
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_disclaimer()

info = load_dataset_info()
pts = info.get("patients", {})
paths = get_paths()

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Patients", str(pts.get("total", 369)), "👥")
with c2:
    metric_card("MRI Model", "SegResNet", "🧠")
with c3:
    metric_card("NLP Model", "BioBERT / ClinicalBERT", "📝")
with c4:
    metric_card("Fusion Accuracy", "87.5% val", "🔗")

c5, c6, c7, c8 = st.columns(4)
with c5:
    metric_card("Clinical Features", "13", "🩺")
with c6:
    metric_card("Segmentation Dice", "~0.85 WT", "📐")
with c7:
    metric_card("GPU Status", get_device_name()[:18], "⚡")
with c8:
    analysis = get_analysis()
    status = "Ready" if analysis.get("analysis_complete") else "Awaiting"
    metric_card("Session", status, "🔬")

st.markdown("### Analysis Pipeline")
pipeline_flow()

st.markdown("### Platform Capabilities")
features = [
    "3D SegResNet tumor segmentation with sub-region masks (WT / TC / ET)",
    "Automatic bounding-box localization with multi-lesion support",
    "BioBERT radiology report embeddings fused with imaging features",
    "Clinical decision support with Low / Medium / High risk stratification",
    "Grad-CAM, SHAP, PCA / t-SNE explainability",
    "PDF / PNG / CSV / JSON / NIfTI export",
]
for f in features:
    st.markdown(f"- {f}")

st.markdown("---")
if st.button("🚀 Start Analysis — MRI Workstation", type="primary", width='stretch'):
    st.switch_page("pages/1_MRI_Analysis.py")

if not paths.segresnet_checkpoint.exists():
    st.error("SegResNet checkpoint missing. Place `best_model.pth` in `models/segmentation/`.")
