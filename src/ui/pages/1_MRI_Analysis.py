"""MRI Analysis — upload, segmentation, bounding boxes, 3D, Grad-CAM."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ui.bootstrap import setup_page
from src.ui.components.mri_viewer import render_gradcam_panel, render_mri_workstation
from src.ui.components.pipeline_status import run_analysis_with_ui
from src.ui.components.plotly_3d import render_3d_tumor_view
from src.ui.utils.disclaimer import render_disclaimer
from src.ui.utils.mri_io import (
    format_transform_error,
    get_stored_upload_paths,
    load_nifti_modalities,
    load_pt_volume,
    save_upload_to_temp,
    store_upload_paths,
)
from src.ui.utils.session import get_analysis, update_analysis

setup_page("MRI Analysis", icon="🖼️")

st.title("MRI Analysis Workstation")
render_disclaimer()

analysis = get_analysis()
UPLOAD_KEY = "cortex_nifti_paths"
PT_KEY = "cortex_pt_path"

tab_upload, tab_view, tab_stats, tab_3d, tab_gradcam = st.tabs(
    ["Upload & Analyze", "Interactive Viewer", "Tumor Statistics", "3D Visualization", "Grad-CAM"]
)

with tab_upload:
    st.subheader("Upload Study")
    upload_type = st.radio("Format", ["Single .pt volume", "Four NIfTI modalities (.nii / .nii.gz)"], horizontal=True)

    report_text = st.text_area(
        "Clinical report (required for fusion)",
        value=analysis.get("report_text") or "",
        height=120,
        placeholder="Paste radiology report text…",
    )

    mri_bundle = analysis.get("mri")

    if upload_type.startswith("Single"):
        pt_file = st.file_uploader("Preprocessed .pt file", type=["pt"])
        if pt_file is not None:
            path = save_upload_to_temp(pt_file, ".pt")
            st.session_state[PT_KEY] = str(path)
            st.caption(f"Saved upload: `{path.name}`")

        if st.button("Load .pt volume", type="secondary"):
            pt_path = st.session_state.get(PT_KEY)
            if not pt_path:
                st.error("Upload a .pt file first.")
            else:
                with st.spinner("Loading preprocessed volume…"):
                    try:
                        bundle = load_pt_volume(pt_path)
                        update_analysis(mri=bundle, patient_id=bundle["patient_id"])
                        mri_bundle = bundle
                        st.success(f"Loaded {bundle['patient_id']} — shape {tuple(bundle['image'].shape)}")
                    except Exception as exc:
                        st.error(f"Failed to load .pt: {exc}")
    else:
        st.caption("Upload all four modalities, then click **Load NIfTI study** (loading is deferred to avoid slow reruns).")
        cols = st.columns(4)
        mods = ["flair", "t1", "t1ce", "t2"]
        pending: dict[str, Path] = {}
        for col, mod in zip(cols, mods):
            with col:
                f = st.file_uploader(mod.upper(), type=["nii", "gz"], key=f"up_{mod}")
                if f is not None:
                    pending[mod] = save_upload_to_temp(f)

        if len(pending) == 4:
            store_upload_paths(UPLOAD_KEY, pending)
            st.success("Four modalities uploaded — ready to load.")

        pid = st.text_input("Patient ID", value=analysis.get("patient_id") or "Uploaded_Patient")

        if st.button("Load NIfTI study", type="secondary"):
            stored = get_stored_upload_paths(UPLOAD_KEY)
            if stored is None:
                st.error("Upload FLAIR, T1, T1CE, and T2 before loading.")
            else:
                with st.spinner("Loading and preprocessing NIfTI (crop + normalize)…"):
                    try:
                        bundle = load_nifti_modalities(stored, patient_id=pid)
                        update_analysis(mri=bundle, patient_id=bundle["patient_id"])
                        mri_bundle = bundle
                        st.success(f"Loaded NIfTI — shape {tuple(bundle['image'].shape)}")
                    except Exception as exc:
                        st.error(format_transform_error(exc))

    if mri_bundle is not None:
        st.info(
            f"MRI ready: **{mri_bundle.get('patient_id', '—')}** — "
            f"tensor shape `{tuple(mri_bundle['image'].shape)}`"
        )

    gradcam_target = st.selectbox("Grad-CAM target class", ["ET", "TC", "WT"])

    if st.button(
        "Run Full Multimodal Analysis",
        type="primary",
        disabled=mri_bundle is None or not report_text.strip(),
    ):
        if mri_bundle and report_text.strip():
            with st.spinner("Running multimodal pipeline…"):
                result = run_analysis_with_ui(mri_bundle, report_text, gradcam_target=gradcam_target)
                update_analysis(**result)
            st.success("Analysis complete.")
            st.rerun()

    if analysis.get("analysis_complete"):
        st.markdown("---")
        stats = analysis.get("stats")
        if stats:
            st.dataframe(
                {"Metric": [r[0] for r in stats.table_rows()], "Value": [r[1] for r in stats.table_rows()]},
                width='stretch',
                hide_index=True,
            )

with tab_view:
    render_mri_workstation(get_analysis())

with tab_stats:
    stats = analysis.get("stats")
    if stats:
        st.subheader("Professional Tumor Statistics")
        st.dataframe(
            {"Metric": [r[0] for r in stats.table_rows()], "Value": [r[1] for r in stats.table_rows()]},
            width='stretch',
            hide_index=True,
        )
        if stats.bboxes:
            st.subheader("Lesion Bounding Boxes")
            for box in stats.bboxes:
                st.markdown(
                    f"**Lesion {box.lesion_id}** — "
                    f"Size {box.width}×{box.height}×{box.depth} vox | "
                    f"Center ({box.center_z:.0f}, {box.center_y:.0f}, {box.center_x:.0f}) | "
                    f"Volume {box.volume_cm3:.2f} cm³ | "
                    f"Largest Ø {box.largest_diameter_mm:.1f} mm"
                )
    else:
        st.info("Run analysis on the Upload tab first.")

with tab_3d:
    render_3d_tumor_view(analysis)

with tab_gradcam:
    render_gradcam_panel(analysis)
