"""
Clinical Feature Engineering for CortexAI Fusion Module.

Implements NB03 logic as a reusable, importable module:

  clean_report()            — whitespace normalization (NO lowercasing)
  extract_report_features() — keyword extraction from radiology reports
  build_clinical_table()    — merge MRI volume stats + report features
  drop_constant_features()  — remove zero-variance columns (fit on train)
  drop_rare_features()      — remove very rare binary columns (fit on train)
  build_all_splits()        — full NB03 pipeline in one call

The dropping lists are computed on TRAIN only and saved as JSON so
validation / test always apply the exact same column set.

Author: Ammar Kamal
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "ANATOMICAL_REGIONS",
    "LATERALITY",
    "RADIOLOGY_FINDINGS",
    "clean_report",
    "contains_any",
    "extract_report_features",
    "build_clinical_table",
    "drop_constant_features",
    "drop_rare_features",
    "build_all_splits",
]

# ── Keyword dictionaries ──────────────────────────────────────────────────────

ANATOMICAL_REGIONS: dict[str, list[str]] = {
    "frontal":         ["frontal"],
    "temporal":        ["temporal"],
    "parietal":        ["parietal"],
    "occipital":       ["occipital"],
    "insula":          ["insula", "insular"],
    "thalamus":        ["thalamus", "thalamic"],
    "basal_ganglia":   ["basal ganglia"],
    "corpus_callosum": ["corpus callosum"],
    "ventricle":       ["ventricle", "ventricular"],
}

LATERALITY: dict[str, list[str]] = {
    "left":     ["left"],
    "right":    ["right"],
    "bilateral": ["bilateral", "both"],
}

RADIOLOGY_FINDINGS: dict[str, list[str]] = {
    "edema":         ["edema"],
    "necrosis":      ["necrosis", "necrotic"],
    "enhancement":   ["enhancement", "enhancing"],
    "mass_effect":   ["mass effect"],
    "midline_shift": ["midline shift"],
    "compression":   ["compression", "compressed"],
    "hemorrhage":    ["hemorrhage", "hemorrhagic"],
}


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_report(text: Any) -> str:
    """
    Whitespace normalization only — NO lowercasing.

    BioBERT and ClinicalBERT are cased models; lowercasing discards
    casing signal they were trained to use. Keyword matching is done
    case-insensitively in contains_any(), not here.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any(text: str, keywords: list[str]) -> int:
    """Return 1 if any keyword appears in text (case-insensitive)."""
    text_lower = text.lower()
    return int(any(kw in text_lower for kw in keywords))


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_report_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract structured binary and numeric features from clean_report column.

    Input dataframe must have columns: patient_id, clean_report.

    Returns a new DataFrame with patient_id + all extracted features.
    """
    rows = []
    for _, row in df.iterrows():
        report = row["clean_report"]
        sample: dict[str, Any] = {"patient_id": row["patient_id"]}

        for feat, kws in ANATOMICAL_REGIONS.items():
            sample[feat] = contains_any(report, kws)

        for feat, kws in LATERALITY.items():
            sample[feat] = contains_any(report, kws)

        for feat, kws in RADIOLOGY_FINDINGS.items():
            sample[feat] = contains_any(report, kws)

        sample["word_count"]     = len(report.split())
        sample["sentence_count"] = report.count(".")
        sample["lobe_count"]     = (
            sample["frontal"] + sample["temporal"]
            + sample["parietal"] + sample["occipital"]
        )
        rows.append(sample)

    return pd.DataFrame(rows)


# ── Column dropping (fit on train, apply to all) ──────────────────────────────

def drop_constant_features(
    train_df:        pd.DataFrame,
    validation_df:   pd.DataFrame,
    test_df:         pd.DataFrame,
    save_dir:        Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Drop columns that are constant in the train split.

    Saves the dropped list to {save_dir}/dropped_constant_features.json
    so it can be reapplied to new data without re-running this function.

    Returns (train, validation, test, dropped_columns).
    """
    constant_cols = [
        c for c in train_df.columns
        if c != "patient_id" and train_df[c].nunique() == 1
    ]
    print(f"Constant features dropped: {constant_cols}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "dropped_constant_features.json").write_text(
            json.dumps(constant_cols)
        )

    return (
        train_df.drop(columns=constant_cols),
        validation_df.drop(columns=constant_cols),
        test_df.drop(columns=constant_cols),
        constant_cols,
    )


def drop_rare_features(
    train_df:      pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df:       pd.DataFrame,
    min_count:     int = 5,
    save_dir:      Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Drop binary columns with fewer than min_count positive examples in train.

    Saves the dropped list to {save_dir}/dropped_rare_features.json.

    Returns (train, validation, test, dropped_columns).
    """
    feature_counts = train_df.drop(columns="patient_id").sum()
    rare_cols = feature_counts[feature_counts < min_count].index.tolist()
    print(f"Rare features dropped (< {min_count} in train): {rare_cols}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "dropped_rare_features.json").write_text(
            json.dumps(rare_cols)
        )

    return (
        train_df.drop(columns=rare_cols),
        validation_df.drop(columns=rare_cols),
        test_df.drop(columns=rare_cols),
        rare_cols,
    )


# ── High-level builder ────────────────────────────────────────────────────────

def build_clinical_table(
    reports_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge metadata (patient_id + MRI volume stats) with report features.

    Parameters
    ----------
    reports_df  : DataFrame with patient_id + report columns
    metadata_df : DataFrame with patient_id + wt_volume / tc_volume / et_volume

    Returns
    -------
    Merged DataFrame (one row per patient).
    """
    reports_df = reports_df[["patient_id", "report"]].copy()
    reports_df["clean_report"] = reports_df["report"].apply(clean_report)

    findings = extract_report_features(reports_df)
    clinical = metadata_df.merge(findings, on="patient_id")

    # report_length is redundant with word_count (high correlation) — drop it
    if "report_length" in clinical.columns:
        clinical = clinical.drop(columns=["report_length"])

    return clinical


def build_all_splits(
    reports_df:        pd.DataFrame,
    train_meta:        pd.DataFrame,
    validation_meta:   pd.DataFrame,
    test_meta:         pd.DataFrame,
    output_dir:        Path,
    min_rare_count:    int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full NB03 pipeline in one call.

    Steps:
        1. Filter reports to each split via patient_id merge
        2. Extract report features
        3. Merge with MRI metadata
        4. Drop constant features (fit on train)
        5. Drop rare features (fit on train)
        6. Save to output_dir/{split}_clinical_features.csv

    Returns (train_clinical, validation_clinical, test_clinical).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _build_one(meta: pd.DataFrame) -> pd.DataFrame:
        merged = reports_df.merge(meta[["patient_id"]], on="patient_id", how="inner")
        return build_clinical_table(merged, meta)

    train_cl = _build_one(train_meta)
    val_cl   = _build_one(validation_meta)
    test_cl  = _build_one(test_meta)

    # Validate no cross-split leakage
    train_ids = set(train_cl["patient_id"])
    val_ids   = set(val_cl["patient_id"])
    test_ids  = set(test_cl["patient_id"])
    assert not (train_ids & val_ids),  "Train/Validation patient overlap!"
    assert not (train_ids & test_ids), "Train/Test patient overlap!"
    assert not (val_ids   & test_ids), "Validation/Test patient overlap!"

    # Drop constant features
    train_findings  = train_cl.drop(columns=[c for c in train_meta.columns if c != "patient_id"], errors="ignore")
    val_findings    = val_cl.drop(columns=[c for c in validation_meta.columns if c != "patient_id"], errors="ignore")
    test_findings   = test_cl.drop(columns=[c for c in test_meta.columns if c != "patient_id"], errors="ignore")

    train_cl, val_cl, test_cl, _ = drop_constant_features(
        train_cl, val_cl, test_cl, save_dir=output_dir
    )
    train_cl, val_cl, test_cl, _ = drop_rare_features(
        train_cl, val_cl, test_cl,
        min_count=min_rare_count,
        save_dir=output_dir,
    )

    # Save
    for split, df in [("train", train_cl), ("validation", val_cl), ("test", test_cl)]:
        path = output_dir / f"{split}_clinical_features.csv"
        df.to_csv(path, index=False)
        print(f"  {split:12}  {df.shape}  → {path}")

    return train_cl, val_cl, test_cl
