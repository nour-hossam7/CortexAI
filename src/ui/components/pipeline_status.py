"""Live pipeline progress UI."""

from __future__ import annotations

import streamlit as st

from src.ui.utils.pipeline import STAGES, PipelineCallbacks, run_full_analysis
from src.ui.utils.session import get_analysis, update_analysis
from src.ui.utils.theme import get_theme


def render_pipeline_status(log: list[str] | None = None) -> None:
    t = get_theme()
    st.markdown("#### Pipeline Status")
    current = log[-1] if log else "Idle"
    for stage in STAGES[:-1]:
        cls = "cortex-pipeline-step"
        if stage == current:
            cls += " cortex-pipeline-active"
        elif log and stage in log:
            cls += " cortex-pipeline-done"
        st.markdown(f"<div class='{cls}'>{stage}</div>", unsafe_allow_html=True)
    if log and "Completed" in log:
        st.markdown(
            f"<div class='cortex-pipeline-step cortex-pipeline-done' style='color:{t['success']}'>✓ Completed</div>",
            unsafe_allow_html=True,
        )


def run_analysis_with_ui(mri_bundle: dict, report_text: str, gradcam_target: str = "ET") -> dict:
    progress = st.progress(0, text="Starting analysis…")
    status = st.empty()
    log_box = st.empty()
    logs: list[str] = []

    def on_stage(name: str, pct: float) -> None:
        progress.progress(min(pct, 1.0), text=name)
        status.markdown(f"**Current stage:** {name}")
        logs.append(name)
        log_box.code("\n".join(logs[-8:]))

    result = run_full_analysis(
        mri_bundle=mri_bundle,
        report_text=report_text,
        gradcam_target=gradcam_target,
        callbacks=PipelineCallbacks(on_stage=on_stage, on_log=lambda m: logs.append(m)),
    )
    progress.progress(1.0, text="Completed")
    update_analysis(**result)
    return result
