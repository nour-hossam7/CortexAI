"""Clinical entity extraction and report highlighting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.fusion_module.clinical_features import (
    ANATOMICAL_REGIONS,
    LATERALITY,
    RADIOLOGY_FINDINGS,
    clean_report,
    extract_report_features,
)


ENTITY_COLORS = {
    "tumor": "#e63946",
    "location": "#2a9d8f",
    "edema": "#457b9d",
    "necrosis": "#6a4c93",
    "enhancement": "#f4a261",
    "mass_effect": "#e76f51",
    "midline_shift": "#d62828",
    "laterality": "#1d3557",
    "lobe": "#264653",
    "hemorrhage": "#9b2226",
    "compression": "#bc6c25",
}


@dataclass
class HighlightSpan:
    start: int
    end: int
    label: str
    keyword: str
    color: str


def extract_clinical_entities(report_text: str) -> dict[str, Any]:
    df = pd.DataFrame([{"patient_id": "current", "report": report_text}])
    df["clean_report"] = df["report"].apply(clean_report)
    features = extract_report_features(df).iloc[0].to_dict()
    text = df["clean_report"].iloc[0]
    spans = _find_highlight_spans(text)
    findings = _build_findings_list(features, text)
    return {
        "clean_report": text,
        "features": features,
        "spans": spans,
        "findings": findings,
    }


def highlight_report_html(report_text: str, entities: dict | None = None) -> str:
    if entities is None:
        entities = extract_clinical_entities(report_text)
    text = entities["clean_report"]
    spans: list[HighlightSpan] = entities["spans"]
    if not spans:
        return f'<p style="line-height:1.7;font-size:15px;">{ _escape(text) }</p>'

    spans = sorted(spans, key=lambda s: s.start)
    parts: list[str] = []
    cursor = 0
    for sp in spans:
        if sp.start < cursor:
            continue
        parts.append(_escape(text[cursor:sp.start]))
        chunk = text[sp.start:sp.end]
        parts.append(
            f'<mark style="background:{sp.color}33;border-bottom:2px solid {sp.color};'
            f'padding:0 2px;border-radius:3px;" title="{sp.label}">{_escape(chunk)}</mark>'
        )
        cursor = sp.end
    parts.append(_escape(text[cursor:]))
    body = "".join(parts)
    return f'<p style="line-height:1.8;font-size:15px;color:#eaeaea;">{body}</p>'


def _find_highlight_spans(text: str) -> list[HighlightSpan]:
    text_lower = text.lower()
    spans: list[HighlightSpan] = []
    keyword_map: list[tuple[str, str, str]] = []
    for region, kws in ANATOMICAL_REGIONS.items():
        for kw in kws:
            keyword_map.append((kw, f"Lobe: {region}", ENTITY_COLORS["lobe"]))
    for lat, kws in LATERALITY.items():
        for kw in kws:
            keyword_map.append((kw, f"Laterality: {lat}", ENTITY_COLORS["laterality"]))
    for finding, kws in RADIOLOGY_FINDINGS.items():
        color = ENTITY_COLORS.get(finding, ENTITY_COLORS["tumor"])
        for kw in kws:
            keyword_map.append((kw, finding.replace("_", " ").title(), color))

    tumor_terms = ["tumor", "glioma", "mass", "lesion", "neoplasm"]
    for kw in tumor_terms:
        keyword_map.append((kw, "Tumor", ENTITY_COLORS["tumor"]))

    for kw, label, color in keyword_map:
        for match in re.finditer(re.escape(kw), text_lower, flags=re.IGNORECASE):
            spans.append(
                HighlightSpan(
                    start=match.start(),
                    end=match.end(),
                    label=label,
                    keyword=kw,
                    color=color,
                )
            )
    return _merge_spans(spans)


def _merge_spans(spans: list[HighlightSpan]) -> list[HighlightSpan]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    merged: list[HighlightSpan] = []
    for sp in spans:
        if merged and sp.start < merged[-1].end:
            continue
        merged.append(sp)
    return merged


def _build_findings_list(features: dict, text: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    t_lower = text.lower()

    if any(w in t_lower for w in ["tumor", "glioma", "mass", "lesion"]):
        findings.append({"category": "Tumor", "detail": "Neoplastic lesion language detected in report.", "severity": "high"})

    for region in ["frontal", "temporal", "parietal", "occipital"]:
        if features.get(region):
            findings.append({"category": "Location", "detail": f"{region.title()} lobe referenced.", "severity": "info"})

    for lat in ["left", "right", "bilateral"]:
        if features.get(lat):
            findings.append({"category": "Laterality", "detail": f"{lat.title()} sided involvement noted.", "severity": "info"})

    mapping = [
        ("edema", "Edema", "Peritumoral edema described."),
        ("necrosis", "Necrosis", "Necrotic component reported."),
        ("enhancement", "Enhancement", "Contrast enhancement described."),
        ("mass_effect", "Mass Effect", "Mass effect on adjacent structures."),
        ("midline_shift", "Midline Shift", "Midline shift reported."),
        ("compression", "Compression", "Compression of adjacent structures."),
        ("hemorrhage", "Hemorrhage", "Hemorrhagic features noted."),
    ]
    for key, cat, detail in mapping:
        if features.get(key):
            findings.append({"category": cat, "detail": detail, "severity": "moderate"})

    size_match = re.search(r"(\d+(?:\.\d+)?)\s*(cm|mm)", text, re.I)
    if size_match:
        findings.append({
            "category": "Size",
            "detail": f"Reported dimension: {size_match.group(0)}.",
            "severity": "info",
        })

    return findings


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
