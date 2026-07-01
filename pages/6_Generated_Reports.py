"""Generated Reports — export PDF, PNG, CSV, JSON, NIfTI."""

from __future__ import annotations

import streamlit as st

from src.ui.bootstrap import setup_page
from src.ui.utils.disclaimer import MEDICAL_DISCLAIMER, render_disclaimer
from src.ui.utils.export import (
    export_csv_stats,
    export_json,
    export_nifti_mask,
    export_pdf_report,
    export_png_slice,
)
from src.ui.utils.session import get_analysis

setup_page("Generated Reports", icon="📄")

st.title("Generated Reports & Export")
render_disclaimer()

analysis = get_analysis()

if not analysis.get("analysis_complete"):
    st.warning("Run full analysis before exporting reports.")
    st.stop()

summary = analysis.get("ai_summary") or {}
if summary:
    st.subheader("AI-Generated Medical Summary")
    st.text(summary.get("full_text", ""))

st.markdown("---")
st.subheader("Download Artifacts")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    pdf = export_pdf_report(analysis)
    if pdf:
        st.download_button("📕 PDF Report", pdf, file_name=f"{analysis['patient_id']}_report.pdf", mime="application/pdf")
    else:
        st.caption("PDF unavailable (install reportlab)")

with c2:
    png = export_png_slice(analysis, view="overlay")
    if png:
        st.download_button("🖼️ PNG Slice", png, file_name=f"{analysis['patient_id']}_overlay.png", mime="image/png")

with c3:
    st.download_button(
        "📊 CSV Stats",
        export_csv_stats(analysis),
        file_name=f"{analysis['patient_id']}_stats.csv",
        mime="text/csv",
    )

with c4:
    st.download_button(
        "📋 JSON Bundle",
        export_json(analysis),
        file_name=f"{analysis['patient_id']}_analysis.json",
        mime="application/json",
    )

with c5:
    nii = export_nifti_mask(analysis)
    if nii:
        st.download_button("🧠 NIfTI Mask", nii, file_name=f"{analysis['patient_id']}_seg.nii.gz", mime="application/gzip")
    else:
        st.caption("NIfTI unavailable")

st.markdown("---")
view = st.selectbox("Preview export slice", ["overlay", "mri", "bbox", "gradcam"])
preview = export_png_slice(analysis, view=view)
if preview:
    st.image(preview, caption=f"Export preview — {view}", width='stretch')

st.caption(MEDICAL_DISCLAIMER)
