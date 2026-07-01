"""Interactive MRI workstation with multi-planar views."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.ui.utils.gradcam_helpers import apply_heatmap_intensity, overlay_heatmap_on_slice
from src.ui.utils.mri_io import get_slice, normalize_slice, tensor_to_numpy
from src.ui.utils.tumor_analysis import draw_bbox_on_slice, mask_to_rgba_overlay


VIEW_AXES = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
MODALITY_LABELS = ["FLAIR", "T1", "T1ce", "T2"]


def render_mri_workstation(analysis: dict, key_prefix: str = "mri") -> None:
    mri = analysis.get("mri")
    mask = analysis.get("mask")
    if mri is None:
        st.info("Upload and analyze an MRI study to use the workstation.")
        return

    image = tensor_to_numpy(mri["image"])
    pred = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
    # predict_mask may return (1, D, H, W) with a batch dim — squeeze it
    # down to exactly (D, H, W) so axis 0/1/2 always index spatial dims.
    while pred.ndim > 3:
        pred = pred[0]
    if pred.ndim < 3:
        st.error(f"Unexpected mask shape {pred.shape} — expected 3D volume.")
        return
    bboxes = analysis.get("bboxes") or []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        view = st.selectbox("Plane", list(VIEW_AXES.keys()), key=f"{key_prefix}_view")
    with c2:
        modality = st.selectbox("Modality", MODALITY_LABELS, key=f"{key_prefix}_mod")
    with c3:
        display_mode = st.selectbox(
            "Display",
            ["Original MRI", "Segmentation", "Overlay", "Bounding Box", "GradCAM"],
            key=f"{key_prefix}_mode",
        )
    with c4:
        gradcam_target = st.selectbox("GradCAM Target", ["ET", "TC", "WT"], key=f"{key_prefix}_gc_target")

    axis = VIEW_AXES[view]
    max_idx = int(pred.shape[axis] - 1)
    slice_idx = st.slider("Slice", 0, int(max_idx), int(max_idx // 2), key=f"{key_prefix}_slice")

    w1, w2, w3 = st.columns(3)
    with w1:
        window_center = st.slider("Window Center", 0.0, 1.0, 0.5, 0.01, key=f"{key_prefix}_wc")
    with w2:
        window_width = st.slider("Window Width", 0.05, 1.0, 0.8, 0.01, key=f"{key_prefix}_ww")
    with w3:
        overlay_alpha = st.slider("Overlay Opacity", 0.0, 1.0, 0.45, 0.05, key=f"{key_prefix}_alpha")

    heat_gain = 1.0
    heat_alpha = 0.5
    if display_mode == "GradCAM":
        g1, g2 = st.columns(2)
        with g1:
            heat_gain = st.slider("Heatmap Intensity", 0.5, 2.5, 1.0, 0.1, key=f"{key_prefix}_hg")
        with g2:
            heat_alpha = st.slider("Heatmap Opacity", 0.1, 1.0, 0.5, 0.05, key=f"{key_prefix}_ha")

    mod_idx = MODALITY_LABELS.index(modality)
    sl = get_slice(image[mod_idx], axis, slice_idx)
    sl = _window_slice(normalize_slice(sl), window_center, window_width)
    msl = get_slice(pred, axis, slice_idx)

    rgb = _render_display(
        sl, msl, bboxes, axis, slice_idx, display_mode,
        analysis, overlay_alpha, gradcam_target, heat_gain, heat_alpha,
    )

    st.image(rgb, caption=f"{view} — {modality} — slice {slice_idx} — {display_mode}", width='stretch')

    if bboxes:
        with st.expander("Bounding Boxes on Current Slice"):
            for box in bboxes:
                visible = _box_visible(box, axis, slice_idx)
                if visible:
                    st.markdown(
                        f"**Lesion #{box.lesion_id}** — "
                        f"W×H×D: {box.width}×{box.height}×{box.depth} vox | "
                        f"Center (z,y,x): ({box.center_z:.1f}, {box.center_y:.1f}, {box.center_x:.1f}) | "
                        f"Vol: {box.volume_cm3:.2f} cm³ | Max Ø: {box.largest_diameter_mm:.1f} mm"
                    )


def _render_display(sl, msl, bboxes, axis, slice_idx, mode, analysis, alpha, gc_target, heat_gain, heat_alpha):
    if mode == "Original MRI":
        return (np.stack([sl] * 3, axis=-1) * 255).astype(np.uint8)

    if mode == "Segmentation":
        seg = np.zeros((*msl.shape, 3))
        colors = {1: [0.9, 0.2, 0.2], 2: [0.2, 0.8, 0.2], 3: [0.2, 0.4, 1.0]}
        for lbl, c in colors.items():
            seg[msl == lbl] = c
        return (seg * 255).astype(np.uint8)

    if mode == "Overlay":
        rgb = mask_to_rgba_overlay(sl, msl, alpha=alpha)
        return rgb

    if mode == "Bounding Box":
        base = mask_to_rgba_overlay(sl, msl, alpha=0.25)
        return draw_bbox_on_slice(base, bboxes, axis, slice_idx, thickness=4)

    if mode == "GradCAM":
        gradcam = analysis.get("gradcam")
        if gradcam is None:
            base = (np.stack([sl] * 3, axis=-1) * 255).astype(np.uint8)
            return draw_bbox_on_slice(base, bboxes, axis, slice_idx)
        hm_sl = get_slice(gradcam, axis, slice_idx)
        hm_sl = apply_heatmap_intensity(hm_sl, gain=heat_gain)
        rgb = overlay_heatmap_on_slice(sl, hm_sl, alpha=heat_alpha)
        return draw_bbox_on_slice(rgb, bboxes, axis, slice_idx)

    return (np.stack([sl] * 3, axis=-1) * 255).astype(np.uint8)


def _window_slice(norm: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = max(0.0, center - width / 2)
    hi = min(1.0, center + width / 2)
    if hi - lo < 1e-6:
        return norm
    out = (norm - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def _box_visible(box, axis, idx) -> bool:
    if axis == 0:
        return box.z_min <= idx <= box.z_max
    if axis == 1:
        return box.y_min <= idx <= box.y_max
    return box.x_min <= idx <= box.x_max


def render_gradcam_panel(analysis: dict) -> None:
    """Side-by-side Original | Segmentation | GradCAM | Bounding Box."""
    if not analysis.get("analysis_complete"):
        st.info("Run full analysis to generate Grad-CAM.")
        return

    mri = analysis.get("mri")
    mask = analysis.get("mask")
    gradcam = analysis.get("gradcam")
    bboxes = analysis.get("bboxes") or []

    image = tensor_to_numpy(mri["image"])
    pred = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
    while pred.ndim > 3:
        pred = pred[0]
    axis = 0
    slice_idx = pred.shape[0] // 2
    sl = normalize_slice(get_slice(image[0], axis, slice_idx))
    msl = get_slice(pred, axis, slice_idx)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption("Original MRI")
        st.image((np.stack([sl] * 3, axis=-1) * 255).astype(np.uint8), width='stretch')
    with c2:
        st.caption("Segmentation")
        st.image(mask_to_rgba_overlay(sl, msl), width='stretch')
    with c3:
        st.caption(f"Grad-CAM ({analysis.get('gradcam_target', 'ET')})")
        if gradcam is not None:
            hm = apply_heatmap_intensity(get_slice(gradcam, axis, slice_idx))
            st.image(overlay_heatmap_on_slice(sl, hm), width='stretch')
        else:
            st.warning("Grad-CAM not available.")
    with c4:
        st.caption("Bounding Box")
        base = mask_to_rgba_overlay(sl, msl, alpha=0.3)
        st.image(draw_bbox_on_slice(base, bboxes, axis, slice_idx, thickness=4), width='stretch')

    o1, o2 = st.columns(2)
    with o1:
        st.slider("Grad-CAM opacity", 0.1, 1.0, 0.5, 0.05, key="gc_panel_alpha")
    with o2:
        st.slider("Heatmap intensity", 0.5, 2.5, 1.0, 0.1, key="gc_panel_gain")
