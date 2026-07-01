"""End-to-end analysis pipeline with live progress tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from src.cv_module.predict import extract_image_features, predict_mask
from src.fusion_module.clinical_features import clean_report, extract_report_features
from src.nlp_module.predict import extract_text_features
from src.ui.utils.clinical_entities import extract_clinical_entities
from src.ui.utils.gradcam_helpers import compute_gradcam_for_target
from src.ui.utils.models_cache import get_biobert_encoder, get_fusion_predictor, get_segresnet
from src.ui.utils.similar_patients import find_similar_patients_for_case
from src.ui.utils.summary import generate_medical_summary
from src.ui.utils.tumor_analysis import compute_tumor_statistics


STAGES = [
    "Loading MRI",
    "Preprocessing",
    "Segmentation",
    "Extracting Image Features",
    "ClinicalBERT Analysis",
    "Fusion Prediction",
    "Generating Explainability",
    "Generating Medical Report",
    "Completed",
]


@dataclass
class PipelineCallbacks:
    on_stage: Callable[[str, float], None] | None = None
    on_log: Callable[[str], None] | None = None


def run_full_analysis(
    mri_bundle: dict[str, Any],
    report_text: str,
    gradcam_target: str = "ET",
    callbacks: PipelineCallbacks | None = None,
) -> dict[str, Any]:
    cb = callbacks or PipelineCallbacks()
    logs: list[str] = []

    def stage(name: str, pct: float) -> None:
        if cb.on_stage:
            cb.on_stage(name, pct)
        logs.append(name)

    def log(msg: str) -> None:
        logs.append(msg)
        if cb.on_log:
            cb.on_log(msg)

    stage("Loading MRI", 0.05)
    image = mri_bundle["image"]
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image).float()
    patient_id = mri_bundle.get("patient_id", "Uploaded_Patient")
    spacing = tuple(mri_bundle.get("spacing_mm", (1.0, 1.0, 1.0)))

    stage("Preprocessing", 0.12)
    sample = {"image": image}
    device = next(get_segresnet().parameters()).device

    stage("Segmentation", 0.25)
    model = get_segresnet()
    mask = predict_mask(model, sample, remap_to_brats=False)
    stats = compute_tumor_statistics(mask, spacing_mm=spacing, patient_id=patient_id)

    stage("Extracting Image Features", 0.38)
    img_feat = extract_image_features(model, image.unsqueeze(0), device)

    stage("ClinicalBERT Analysis", 0.52)
    tokenizer, enc_model, enc_device = get_biobert_encoder()
    text_feat = extract_text_features(
        report_text,
        tokenizer=tokenizer,
        model=enc_model,
        device=enc_device,
    )
    entities = extract_clinical_entities(report_text)

    clinical_row = _build_clinical_row(report_text, stats)
    fusion_result = None
    unified_repr = None
    shap_importance = _load_shap_importance()

    stage("Fusion Prediction", 0.65)
    predictor = get_fusion_predictor()
    if predictor is not None:
        try:
            fusion_result = predictor.predict(
                img_feat, text_feat, clinical_row, patient_id=patient_id
            )
            unified_repr = _extract_unified_repr(predictor, img_feat, text_feat, clinical_row)
        except Exception as exc:
            log(f"Fusion inference skipped: {exc}")
    else:
        log("Fusion checkpoint unavailable — prediction skipped.")

    stage("Generating Explainability", 0.78)
    gradcam = None
    try:
        gradcam = compute_gradcam_for_target(model, image, target=gradcam_target)
    except Exception as exc:
        log(f"Grad-CAM skipped: {exc}")

    similar = find_similar_patients_for_case(
        patient_id=patient_id,
        unified_repr=unified_repr,
        image_features=img_feat,
        stats=stats,
        fusion=fusion_result,
    )

    stage("Generating Medical Report", 0.92)
    ai_summary = generate_medical_summary(
        patient_id=patient_id,
        stats=stats,
        entities=entities,
        fusion=fusion_result,
        shap_importance=shap_importance,
        report_excerpt=entities["clean_report"][:500],
    )

    stage("Completed", 1.0)

    return {
        "patient_id": patient_id,
        "mri": mri_bundle,
        "mask": mask,
        "report_text": report_text,
        "stats": stats,
        "bboxes": stats.bboxes,
        "lesions": stats.bboxes,
        "image_features": img_feat,
        "text_features": text_feat,
        "clinical_row": clinical_row,
        "clinical_entities": entities,
        "fusion": fusion_result,
        "unified_repr": unified_repr,
        "gradcam": gradcam,
        "gradcam_target": gradcam_target,
        "shap_importance": shap_importance,
        "ai_summary": ai_summary,
        "similar_patients": similar,
        "pipeline_log": logs,
        "analysis_complete": True,
    }


def _build_clinical_row(report_text: str, stats) -> dict[str, float]:
    df = pd.DataFrame([{"patient_id": stats.patient_id, "report": report_text}])
    df["clean_report"] = df["report"].apply(clean_report)
    findings = extract_report_features(df)
    row = findings.iloc[0].to_dict()
    row["wt_volume"] = float(stats.wt_voxels)
    row["tc_volume"] = float(stats.tc_voxels)
    row["et_volume"] = float(stats.et_voxels)
    return {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in row.items() if k != "patient_id"}


def _extract_unified_repr(predictor, img_feat, txt_feat, clinical_row) -> np.ndarray | None:
    try:
        import torch

        img_t = torch.as_tensor(img_feat, dtype=torch.float32).unsqueeze(0).to(predictor.device)
        txt_t = torch.as_tensor(txt_feat, dtype=torch.float32).unsqueeze(0).to(predictor.device)
        clin_t = predictor._prepare_clinical(clinical_row)
        with torch.no_grad():
            _, unified = predictor.model(img_t, txt_t, clin_t)
        return unified.squeeze(0).cpu().numpy()
    except Exception:
        return None


def _load_shap_importance() -> pd.DataFrame | None:
    from src.ui.utils.paths import get_paths

    path = get_paths().fusion_results / "shap_clinical_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None
