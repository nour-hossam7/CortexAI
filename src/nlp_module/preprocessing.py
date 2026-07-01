"""
Text preprocessing for TextBraTS radiology reports.

Provides:

- Deterministic seed setup
- Report cleaning (whitespace/newline normalization only)
- Applying the CV module's patient-level split to NLP reports

Validated end-to-end in Notebooks 03 (apply-cv-split-to-textbrats) and
04 (medical-text-preprocessing).

Author:
Nour Hossam
"""

import json
import random
from pathlib import Path
import re
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config

__all__ = [
    "set_seed",
    "clean_report",
    "load_dataset_split",
    "load_report_inventory",
    "split_reports_by_patient",
]


def set_seed() -> None:
    """
    Set deterministic behavior for any randomness used by this module
    (none currently — kept for interface parity with cv_module.preprocessing
    and in case future steps, e.g. embedding-based sampling, need it).
    """

    random.seed(Config.SEED)
    np.random.seed(Config.SEED)


def clean_report(text: object) -> str:
    """
    Normalize whitespace in a single radiology report.

    Replaces tabs/carriage-returns/newlines with single spaces and
    collapses repeated whitespace, then strips leading/trailing space.

    Deliberately does NOT lowercase, remove stopwords, or stem — BioBERT
    and ClinicalBERT are cased models pretrained on raw clinical text;
    aggressive cleaning would discard signal they were trained to use.
    Verified in Notebook 04 to leave word counts unchanged for all 369
    reports (cleaning only touches whitespace, never content).

    Parameters
    ----------
    text : object
        Raw report text. Coerced to str first so this is safe to call
        on a value that came back as NaN/float from a CSV read.

    Returns
    -------
    str
        Cleaned report text.
    """

    text = str(text)
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_dataset_split() -> Dict[str, List[str]]:
    """
    Load the fixed train/validation/test split from dataset_split.json.

    This is the SAME split file produced by the CV module's
    Notebook 03.5 — not a separately-generated one. Loading it directly
    here (rather than maintaining a second copy) is what guarantees
    patient-level consistency between the CV and NLP modules.
    """

    with open(Config.SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    return split


def load_report_inventory(inventory_path: Path) -> pd.DataFrame:
    """
    Load the full (unsplit) report inventory produced by Notebook 01
    (textbrats_dataset_inventory.csv) — one row per patient, with at
    least a "patient_id" and "report" column.

    Parameters
    ----------
    inventory_path : Path
        Path to textbrats_dataset_inventory.csv.

    Returns
    -------
    pd.DataFrame
    """

    return pd.read_csv(inventory_path)


def split_reports_by_patient(
    dataset_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Apply the CV module's patient-level split to a report inventory
    DataFrame and clean each report.

    Matches Notebooks 03 + 04 exactly: split by patient_id membership,
    verify full coverage and zero overlap between subsets, then apply
    clean_report() to produce a "clean_report" column.

    Parameters
    ----------
    dataset_df : pd.DataFrame
        Must contain "patient_id" and "report" columns — e.g. the
        output of load_report_inventory().

    Returns
    -------
    dict
        {"train": df, "validation": df, "test": df}, each with an
        added "clean_report" column. Row order within each split
        matches dataset_df's original order (filtered, not reordered).

    Raises
    ------
    AssertionError
        If any patient in the split file is missing from dataset_df,
        or if the three subsets are not mutually disjoint.
    """

    split = load_dataset_split()

    subsets: Dict[str, pd.DataFrame] = {}

    all_split_ids = set()
    for name in ("train", "validation", "test"):
        all_split_ids |= set(split[name])

    dataset_ids = set(dataset_df["patient_id"])
    missing = all_split_ids - dataset_ids
    assert not missing, (
        f"{len(missing)} patients in dataset_split.json are missing "
        f"from the report inventory: {sorted(missing)[:5]}..."
    )

    seen_so_far: set = set()

    for name in ("train", "validation", "test"):
        ids = set(split[name])

        assert ids.isdisjoint(seen_so_far), (
            f"Overlap detected: subset '{name}' shares patient IDs "
            f"with an earlier subset — this would leak patients across "
            f"the train/validation/test boundary."
        )
        seen_so_far |= ids

        subset_df = dataset_df[
            dataset_df["patient_id"].isin(ids)
        ].copy()

        subset_df["clean_report"] = subset_df["report"].apply(clean_report)

        subsets[name] = subset_df

    return subsets
