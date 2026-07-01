"""Plotly 3D tumor and brain visualization."""

from __future__ import annotations

import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.ui.utils.mri_io import tensor_to_numpy


def render_3d_tumor_view(analysis: dict) -> None:
    if not HAS_PLOTLY:
        st.warning("Plotly is required for 3D visualization.")
        return

    mri = analysis.get("mri")
    mask = analysis.get("mask")
    if mri is None or mask is None:
        st.info("Run analysis to enable 3D view.")
        return

    pred = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
    image = tensor_to_numpy(mri["image"])
    bboxes = analysis.get("bboxes") or []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_brain = st.checkbox("Brain envelope", True)
    with c2:
        show_tumor = st.checkbox("Tumor surface", True)
    with c3:
        show_seg = st.checkbox("Segmentation voxels", False)
    with c4:
        show_bbox = st.checkbox("Bounding boxes", True)

    tumor_alpha = st.slider("Tumor transparency", 0.05, 1.0, 0.55, 0.05)
    brain_alpha = st.slider("Brain transparency", 0.02, 0.5, 0.12, 0.02)

    fig = go.Figure()

    if show_brain:
        _add_brain_mesh(fig, image[0], brain_alpha)

    wt = (pred > 0).astype(np.uint8)
    if show_tumor and wt.any():
        _add_mask_mesh(fig, wt, color="crimson", opacity=tumor_alpha, name="Whole Tumor")

    if show_seg:
        for lbl, color, name in [(1, "orange", "NCR"), (2, "limegreen", "Edema"), (3, "royalblue", "ET")]:
            m = pred == lbl
            if m.any():
                _add_mask_mesh(fig, m, color=color, opacity=0.35, name=name)

    if show_bbox and bboxes:
        for box in bboxes:
            _add_bbox_wireframe(fig, box)

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        title="3D Tumor Visualization",
    )
    st.plotly_chart(fig, width='stretch')


def _add_brain_mesh(fig, volume: np.ndarray, opacity: float) -> None:
    from skimage import measure

    smoothed = volume.astype(np.float32)
    thresh = np.percentile(smoothed[smoothed > 0], 35) if (smoothed > 0).any() else 0.1
    try:
        verts, faces, _, _ = measure.marching_cubes(smoothed, level=thresh, step_size=3)
        fig.add_trace(go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="lightgray", opacity=opacity, name="Brain",
        ))
    except Exception:
        pass


def _add_mask_mesh(fig, mask: np.ndarray, color: str, opacity: float, name: str) -> None:
    from skimage import measure

    if mask.sum() < 10:
        return
    try:
        verts, faces, _, _ = measure.marching_cubes(mask.astype(float), level=0.5, step_size=2)
        fig.add_trace(go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=opacity, name=name,
        ))
    except Exception:
        coords = np.argwhere(mask)
        step = max(1, len(coords) // 800)
        fig.add_trace(go.Scatter3d(
            x=coords[::step, 2], y=coords[::step, 1], z=coords[::step, 0],
            mode="markers",
            marker=dict(size=2, color=color, opacity=opacity),
            name=name,
        ))


def _add_bbox_wireframe(fig, box) -> None:
    x0, x1 = box.x_min, box.x_max
    y0, y1 = box.y_min, box.y_max
    z0, z1 = box.z_min, box.z_max
    edges = [
        (x0, y0, z0, x1, y0, z0), (x1, y0, z0, x1, y1, z0), (x1, y1, z0, x0, y1, z0), (x0, y1, z0, x0, y0, z0),
        (x0, y0, z1, x1, y0, z1), (x1, y0, z1, x1, y1, z1), (x1, y1, z1, x0, y1, z1), (x0, y1, z1, x0, y0, z1),
        (x0, y0, z0, x0, y0, z1), (x1, y0, z0, x1, y0, z1), (x1, y1, z0, x1, y1, z1), (x0, y1, z0, x0, y1, z1),
    ]
    for ex in edges:
        fig.add_trace(go.Scatter3d(
            x=[ex[0], ex[3]], y=[ex[1], ex[4]], z=[ex[2], ex[5]],
            mode="lines", line=dict(color="red", width=4),
            showlegend=False,
        ))
