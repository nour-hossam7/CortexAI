"""Clinical Report — NLP analysis and entity highlighting."""

from __future__ import annotations

import streamlit as st

from src.ui.bootstrap import setup_page
from src.ui.utils.clinical_entities import ENTITY_COLORS, extract_clinical_entities, highlight_report_html
from src.ui.utils.disclaimer import render_disclaimer
from src.ui.utils.session import get_analysis, update_analysis

setup_page("Clinical Report", icon="📋")

st.title("Clinical Report Analysis")
render_disclaimer()

analysis = get_analysis()

col_l, col_r = st.columns([1, 1])

with col_l:
    st.subheader("Report Input")
    source = st.radio("Input method", ["Paste text", "Upload .txt"], horizontal=True)
    report_text = analysis.get("report_text") or ""

    if source == "Paste text":
        report_text = st.text_area("Radiology report", value=report_text, height=280)
    else:
        txt = st.file_uploader("Report file", type=["txt"])
        if txt:
            report_text = txt.getvalue().decode("utf-8", errors="replace")
            st.text_area("Preview", report_text, height=200)

    if st.button("Analyze Report", type="primary") and report_text.strip():
        entities = extract_clinical_entities(report_text)
        update_analysis(report_text=report_text, clinical_entities=entities)
        if analysis.get("mri") and not analysis.get("analysis_complete"):
            st.info("MRI loaded — run full analysis from MRI page for fusion.")
        st.rerun()

with col_r:
    st.subheader("Highlighted Report")
    if report_text.strip():
        entities = analysis.get("clinical_entities") or extract_clinical_entities(report_text)
        st.markdown(highlight_report_html(report_text, entities), unsafe_allow_html=True)

        st.subheader("Extracted Findings")
        for finding in entities.get("findings", []):
            sev = finding.get("severity", "info")
            color = {"high": "#e63946", "moderate": "#e9c46a", "info": "#4cc9f0"}.get(sev, "#9aa0a6")
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:0.5rem 1rem;margin:0.4rem 0;"
                f"background:rgba(255,255,255,0.03);border-radius:6px;'>"
                f"<strong>{finding['category']}</strong> — {finding['detail']}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Enter or upload a clinical report.")

st.markdown("---")
st.subheader("Entity Legend")
legend_cols = st.columns(4)
for i, (name, color) in enumerate(list(ENTITY_COLORS.items())[:8]):
    with legend_cols[i % 4]:
        st.markdown(f"<span style='color:{color}'>●</span> {name.replace('_', ' ').title()}", unsafe_allow_html=True)

if analysis.get("ai_summary"):
    st.markdown("---")
    st.subheader("AI-Generated Medical Summary")
    summary = analysis["ai_summary"]
    for section in ["Clinical Findings", "Tumor Characteristics", "AI Risk Assessment"]:
        if summary.get(section):
            st.markdown(f"**{section}**")
            st.write(summary[section])
