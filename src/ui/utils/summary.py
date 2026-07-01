"""Dynamic radiology-style summary assembled from model outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_medical_summary(
    patient_id: str,
    stats: Any,
    entities: dict,
    fusion: Any | None,
    shap_importance: pd.DataFrame | None,
    report_excerpt: str = "",
) -> dict[str, str]:
    features = entities.get("features", {})
    findings = entities.get("findings", [])

    clinical_lines = _clinical_findings(findings, features, report_excerpt)
    tumor_lines = _tumor_characteristics(stats)
    risk_lines = _risk_assessment(fusion)
    explain_lines = _model_explanation(shap_importance, fusion, stats)
    followup_lines = _follow_up(fusion, stats)

    return {
        "Clinical Findings": "\n".join(clinical_lines),
        "Tumor Characteristics": "\n".join(tumor_lines),
        "AI Risk Assessment": "\n".join(risk_lines),
        "Model Explanation": "\n".join(explain_lines),
        "Recommended Follow-up": "\n".join(followup_lines),
        "full_text": _assemble_full(patient_id, clinical_lines, tumor_lines, risk_lines, explain_lines, followup_lines),
    }


def _clinical_findings(findings, features, excerpt) -> list[str]:
    lines: list[str] = []
    if excerpt.strip():
        snippet = excerpt.strip()[:280] + ("…" if len(excerpt.strip()) > 280 else "")
        lines.append(f"Source report excerpt: \"{snippet}\"")

    if not findings:
        lines.append("Structured NLP analysis did not extract discrete radiology entities from the supplied text.")
        return lines

    by_cat: dict[str, list[str]] = {}
    for f in findings:
        by_cat.setdefault(f["category"], []).append(f["detail"])
    for cat, details in by_cat.items():
        lines.append(f"{cat}: {'; '.join(details)}.")
    return lines


def _tumor_characteristics(stats) -> list[str]:
    if stats is None or stats.wt_voxels == 0:
        return ["Automated segmentation did not identify a measurable whole-tumor volume in the uploaded study."]

    lines = [
        f"Whole-tumor volume measures {stats.wt_volume_cm3:.2f} cm³ ({stats.wt_voxels:,} voxels).",
        f"Tumor core volume: {stats.tc_volume_cm3:.2f} cm³; enhancing component: {stats.et_volume_cm3:.2f} cm³.",
        f"Maximum diameter estimate: {stats.largest_diameter_mm:.1f} mm; equivalent spherical diameter: {stats.equivalent_diameter_mm:.1f} mm.",
        f"Lesion occupies approximately {stats.tumor_percentage:.2f}% of intracranial voxels on the processed volume.",
        f"Spatial localization: {stats.tumor_laterality}, {stats.tumor_lobe}.",
        f"Largest cross-sectional area on {stats.largest_slice_axis} slice #{stats.largest_slice_index}: {stats.wt_area_mm2:.0f} mm².",
    ]
    if stats.num_lesions > 1:
        lines.append(f"{stats.num_lesions} disconnected enhancing/necrotic components detected — multifocal pattern.")
    else:
        lines.append("Single dominant connected tumor component identified.")
    return lines


def _risk_assessment(fusion) -> list[str]:
    if fusion is None:
        return ["Multimodal fusion model unavailable; risk stratification not computed for this session."]

    probs = fusion.probabilities
    lines = [
        f"Fusion classifier output: {fusion.predicted_label} (confidence {fusion.confidence:.1%}).",
        f"Calibrated class probabilities — Low: {probs.get('Low Risk', 0):.1%}, "
        f"Medium: {probs.get('Medium Risk', 0):.1%}, High: {probs.get('High Risk', 0):.1%}.",
    ]
    top = max(probs.items(), key=lambda x: x[1])
    margin = top[1] - sorted(probs.values())[-2] if len(probs) > 1 else top[1]
    if margin < 0.15:
        lines.append("Probability margin between competing risk classes is narrow — interpret with caution.")
    elif fusion.confidence >= 0.8:
        lines.append("Model confidence exceeds 80% for the predicted risk stratum.")
    return lines


def _model_explanation(shap_importance, fusion, stats) -> list[str]:
    lines: list[str] = []
    if shap_importance is not None and len(shap_importance):
        top = shap_importance.sort_values("importance", ascending=False).head(5)
        drivers = ", ".join(f"{r.feature} ({r.importance:.3f})" for r in top.itertuples())
        lines.append(f"Pre-computed SHAP analysis ranks these clinical-imaging drivers highest: {drivers}.")
    else:
        lines.append("SHAP attribution maps not loaded; explanation based on segmentation metrics and NLP features.")

    if stats and stats.wt_voxels:
        lines.append(
            f"Imaging severity signals include WT volume {stats.wt_volume_cm3:.1f} cm³ and ET volume {stats.et_volume_cm3:.1f} cm³."
        )
    if fusion:
        lines.append(
            f"Decision boundary favored {fusion.predicted_label} given fused 256-d representation + scaled clinical vector."
        )
    return lines


def _follow_up(fusion, stats) -> list[str]:
    lines: list[str] = []
    if fusion and fusion.predicted_label == "High Risk":
        lines.append("Consider expedited neuro-oncology review and contrast-enhanced follow-up MRI per institutional protocol.")
        lines.append("Multidisciplinary tumor board discussion recommended given high-risk classifier output.")
    elif fusion and fusion.predicted_label == "Medium Risk":
        lines.append("Short-interval MRI surveillance (6–8 weeks) may be appropriate if clinically indicated.")
    elif fusion and fusion.predicted_label == "Low Risk":
        lines.append("Routine surveillance interval may be acceptable pending correlation with full clinical context.")
    else:
        lines.append("Clinical correlation and specialist review remain essential when automated risk scoring is unavailable.")

    lines.append("All recommendations are AI-generated decision support only and require physician validation.")
    return lines


def _assemble_full(patient_id, *sections) -> str:
    parts = [f"AI RADIOLOGY SUMMARY — Patient {patient_id}", ""]
    titles = ["CLINICAL FINDINGS", "TUMOR CHARACTERISTICS", "AI RISK ASSESSMENT", "MODEL EXPLANATION", "RECOMMENDED FOLLOW-UP"]
    for title, lines in zip(titles, sections):
        parts.append(title)
        parts.append("-" * len(title))
        parts.extend(lines)
        parts.append("")
    return "\n".join(parts)
