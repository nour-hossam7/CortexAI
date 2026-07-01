"""Export analysis results to PDF, PNG, CSV, JSON, and NIfTI."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from io import BytesIO
from os import unlink
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ui.utils.disclaimer import MEDICAL_DISCLAIMER
from src.ui.utils.mri_io import get_slice, normalize_slice, tensor_to_numpy
from src.ui.utils.tumor_analysis import draw_bbox_on_slice, mask_to_rgba_overlay


def build_export_bundle(analysis: dict[str, Any]) -> dict[str, Any]:
    stats = analysis.get("stats")
    fusion = analysis.get("fusion")
    bundle = {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "patient_id": analysis.get("patient_id"),
        "disclaimer": MEDICAL_DISCLAIMER,
        "tumor_statistics": stats.to_dict() if stats else None,
        "clinical_entities": {
            "findings": analysis.get("clinical_entities", {}).get("findings", []),
            "features": analysis.get("clinical_entities", {}).get("features", {}),
        },
        "fusion_prediction": fusion.to_dict() if fusion else None,
        "ai_summary": analysis.get("ai_summary"),
        "similar_patients": analysis.get("similar_patients").to_dict(orient="records")
        if isinstance(analysis.get("similar_patients"), pd.DataFrame)
        else None,
        "pipeline_log": analysis.get("pipeline_log", []),
    }
    return bundle


def export_json(analysis: dict[str, Any]) -> bytes:
    return json.dumps(build_export_bundle(analysis), indent=2, default=str).encode("utf-8")


def export_csv_stats(analysis: dict[str, Any]) -> bytes:
    stats = analysis.get("stats")
    if stats is None:
        return b"metric,value\n"
    rows = stats.table_rows()
    df = pd.DataFrame(rows, columns=["metric", "value"])
    return df.to_csv(index=False).encode("utf-8")


def export_png_slice(analysis: dict[str, Any], view: str = "overlay") -> bytes | None:
    import matplotlib.pyplot as plt

    mri = analysis.get("mri")
    mask = analysis.get("mask")
    if mri is None or mask is None:
        return None

    image = tensor_to_numpy(mri["image"])
    pred = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
    modality = 0
    axis = 0
    idx = pred.shape[0] // 2
    sl = get_slice(image[modality], axis, idx)
    sl = normalize_slice(sl)
    msl = get_slice(pred, axis, idx)

    if view == "mri":
        rgb = np.stack([sl] * 3, axis=-1)
    elif view == "bbox":
        rgb = mask_to_rgba_overlay(sl, msl) / 255.0
        rgb = draw_bbox_on_slice(
            (rgb * 255).astype(np.uint8),
            analysis.get("bboxes") or [],
            axis,
            idx,
        )
    elif view == "gradcam" and analysis.get("gradcam") is not None:
        from src.ui.utils.gradcam_helpers import overlay_heatmap_on_slice

        hm = get_slice(analysis["gradcam"], axis, idx)
        rgb = overlay_heatmap_on_slice(sl, hm)
    else:
        rgb = mask_to_rgba_overlay(sl, msl)
        rgb = draw_bbox_on_slice(rgb, analysis.get("bboxes") or [], axis, idx)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    ax.axis("off")
    ax.set_title(f"{analysis.get('patient_id')} — {view}")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def export_nifti_mask(analysis: dict[str, Any]) -> bytes | None:
    try:
        import nibabel as nib
    except ImportError:
        return None

    mask = analysis.get("mask")
    if mask is None:
        return None
    arr = mask.numpy().astype(np.uint8) if hasattr(mask, "numpy") else np.asarray(mask, dtype=np.uint8)
    spacing = analysis.get("mri", {}).get("spacing_mm", (1.0, 1.0, 1.0))
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nii = nib.Nifti1Image(arr, affine)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = tmp.name
        nib.save(nii, tmp_path)
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        if tmp_path is not None:
            try:
                unlink(tmp_path)
            except OSError:
                pass


def export_pdf_report(analysis: dict[str, Any]) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return None

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=18, spaceAfter=12)
    story.append(Paragraph("CortexAI — Clinical Decision Support Report", title_style))
    story.append(Paragraph(MEDICAL_DISCLAIMER, styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Patient: {analysis.get('patient_id', 'N/A')}", styles["Heading2"]))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))

    png = export_png_slice(analysis, view="overlay")
    if png:
        story.append(Image(BytesIO(png), width=4 * inch, height=4 * inch))
        story.append(Spacer(1, 0.15 * inch))

    stats = analysis.get("stats")
    if stats:
        data = [["Metric", "Value"]] + stats.table_rows()
        tbl = Table(data, colWidths=[3.2 * inch, 2.8 * inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        story.append(Paragraph("Tumor Statistics", styles["Heading2"]))
        story.append(tbl)
        story.append(Spacer(1, 0.15 * inch))

    fusion = analysis.get("fusion")
    if fusion:
        story.append(Paragraph("Fusion Risk Assessment", styles["Heading2"]))
        story.append(Paragraph(
            f"Prediction: {fusion.predicted_label} — Confidence: {fusion.confidence:.1%}",
            styles["Normal"],
        ))
        for label, prob in fusion.probabilities.items():
            story.append(Paragraph(f"  • {label}: {prob:.1%}", styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))

    summary = analysis.get("ai_summary") or {}
    for section in ["Clinical Findings", "Tumor Characteristics", "AI Risk Assessment", "Model Explanation", "Recommended Follow-up"]:
        if summary.get(section):
            story.append(Paragraph(section, styles["Heading2"]))
            for line in summary[section].split("\n"):
                story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    return buf.getvalue()
