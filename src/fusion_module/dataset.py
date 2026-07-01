"""
Dataset utilities for the CortexAI Fusion Module.

Covers the logic from NB01 (data integration) and NB03 (clinical features):

  load_cv_features()          — load bottleneck_features_{split}.npy
  load_nlp_embeddings()       — load {split}_embeddings.npy
  verify_alignment()          — assert same patient IDs across modalities
  build_fusion_dataset()      — sort + align + combine into one dict per split
  save_fusion_dataset()       — write {split}_fusion.npz + {split}_metadata.csv
  load_fusion_split()         — read a saved {split}_fusion.npz
  FusionDataset               — PyTorch Dataset used by all DataLoaders

Author: Ammar Kamal
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import FusionConfig

__all__ = [
    "FusionDataset",
    "load_cv_features",
    "load_nlp_embeddings",
    "verify_alignment",
    "build_fusion_dataset",
    "save_fusion_dataset",
    "load_fusion_split",
]

SPLITS = ("train", "validation", "test")


# ── CV & NLP loaders ─────────────────────────────────────────────────────────

def load_cv_features(
    cv_results_dir: Path,
) -> dict[str, dict[str, Any]]:
    """
    Load SegResNet bottleneck features for all splits.

    Expects files produced by extract_all_features.py:
        {cv_results_dir}/{split}/bottleneck_features.npy
        {cv_results_dir}/{split}/feature_metadata.csv

    Returns
    -------
    dict  split → {"features": np.ndarray (N,256), "metadata": pd.DataFrame}
    """
    cv: dict = {}
    for split in SPLITS:
        split_dir = cv_results_dir / split
        features_path = split_dir / "bottleneck_features.npy"
        metadata_path = split_dir / "feature_metadata.csv"

        if not features_path.exists():
            raise FileNotFoundError(
                f"CV features not found: {features_path}\n"
                "Run extract_all_features.py first."
            )

        cv[split] = {
            "features": np.load(features_path),
            "metadata": pd.read_csv(metadata_path),
        }
    return cv


def load_nlp_embeddings(
    nlp_dir: Path,
) -> dict[str, dict[str, Any]]:
    """
    Load NLP text embeddings for all splits.

    Expects files produced by the NLP module pipeline:
        {nlp_dir}/{split}_embeddings.npy
        {nlp_dir}/{split}_metadata.csv

    Returns
    -------
    dict  split → {"embeddings": np.ndarray (N,768), "metadata": pd.DataFrame}
    """
    nlp: dict = {}
    for split in SPLITS:
        emb_path  = nlp_dir / f"{split}_embeddings.npy"
        meta_path = nlp_dir / f"{split}_metadata.csv"

        if not emb_path.exists():
            raise FileNotFoundError(
                f"NLP embeddings not found: {emb_path}\n"
                "Run the NLP pipeline first."
            )

        nlp[split] = {
            "embeddings": np.load(emb_path),
            "metadata":   pd.read_csv(meta_path),
        }
    return nlp


# ── Alignment ─────────────────────────────────────────────────────────────────

def verify_alignment(
    cv_data:    dict,
    nlp_data:   dict,
    model_name: str = "",
) -> None:
    """
    Assert that CV and NLP have the same patient IDs in every split.

    Comparison is set-based (order-independent) — actual row alignment
    is fixed by build_fusion_dataset() which sorts both arrays by
    patient_id before combining them.

    Raises
    ------
    AssertionError if any split has a patient set mismatch.
    """
    label = model_name.upper() or "NLP"
    print("=" * 70)
    print(f"Alignment check — {label}")
    print("=" * 70)

    for split in SPLITS:
        cv_ids  = set(cv_data[split]["metadata"]["patient_id"].tolist())
        nlp_ids = set(nlp_data[split]["metadata"]["patient_id"].tolist())

        print(f"\n{split.upper()}")
        print(f"  CV patients  : {len(cv_ids)}")
        print(f"  NLP patients : {len(nlp_ids)}")

        assert cv_ids == nlp_ids, (
            f"{split} alignment failed — patient sets do not match.\n"
            f"  In CV only  : {cv_ids - nlp_ids}\n"
            f"  In NLP only : {nlp_ids - cv_ids}"
        )
        print("  ✓ Patient sets match")


# ── Build & save ──────────────────────────────────────────────────────────────

def build_fusion_dataset(
    cv_data:  dict,
    nlp_data: dict,
) -> dict[str, dict[str, Any]]:
    """
    Sort both feature arrays by patient_id and combine into one dict.

    Row order in bottleneck_features_{split}.npy is NOT guaranteed to
    match row order in {split}_embeddings.npy. Sorting canonicalizes
    both so row i in image_features always corresponds to row i in
    text_features.

    Returns
    -------
    dict  split → {
        "image_features": np.ndarray (N, 256),
        "text_features":  np.ndarray (N, 768),
        "metadata":       pd.DataFrame  sorted by patient_id,
    }
    """
    fusion: dict = {}
    for split in SPLITS:
        cv_meta  = cv_data[split]["metadata"].copy()
        nlp_meta = nlp_data[split]["metadata"].copy()

        cv_sort_idx  = cv_meta.sort_values("patient_id").index.values
        nlp_sort_idx = nlp_meta.sort_values("patient_id").index.values

        fusion[split] = {
            "image_features": cv_data[split]["features"][cv_sort_idx],
            "text_features":  nlp_data[split]["embeddings"][nlp_sort_idx],
            "metadata":       cv_meta.sort_values("patient_id").reset_index(drop=True),
        }
    return fusion


def save_fusion_dataset(
    fusion_data: dict,
    output_dir:  Path,
    encoder_name: str,
) -> None:
    """
    Save fusion dataset to {output_dir}/{encoder_name}/{split}_fusion.npz
    and matching {split}_metadata.csv.
    """
    save_dir = output_dir / encoder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        np.savez_compressed(
            save_dir / f"{split}_fusion.npz",
            image_features=fusion_data[split]["image_features"],
            text_features=fusion_data[split]["text_features"],
        )
        fusion_data[split]["metadata"].to_csv(
            save_dir / f"{split}_metadata.csv",
            index=False,
        )

    print(f"Fusion dataset saved → {save_dir}")
    for split in SPLITS:
        img = fusion_data[split]["image_features"].shape
        txt = fusion_data[split]["text_features"].shape
        n   = len(fusion_data[split]["metadata"])
        print(f"  {split:12}  image {img}  text {txt}  patients {n}")


def load_fusion_split(
    fusion_dir: Path,
    split:      str,
) -> dict[str, Any]:
    """
    Load one saved fusion split.

    Returns
    -------
    dict with keys: image_features, text_features, metadata
    """
    npz  = np.load(fusion_dir / f"{split}_fusion.npz")
    meta = pd.read_csv(fusion_dir / f"{split}_metadata.csv")
    return {
        "image_features": npz["image_features"],
        "text_features":  npz["text_features"],
        "metadata":       meta,
    }


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class FusionDataset(Dataset):
    """
    PyTorch Dataset for multimodal fusion training / evaluation.

    Each item is a dict:
        image    : Tensor (256,)
        text     : Tensor (768,)
        clinical : Tensor (clinical_dim,)
        label    : Tensor scalar int64
        patient_id : str

    Parameters
    ----------
    image_features    : np.ndarray (N, 256)
    text_features     : np.ndarray (N, 768)
    clinical_features : np.ndarray (N, clinical_dim)
    labels            : np.ndarray (N,)  int64
    patient_ids       : list[str]  length N  (optional)
    """

    def __init__(
        self,
        image_features:    np.ndarray,
        text_features:     np.ndarray,
        clinical_features: np.ndarray,
        labels:            np.ndarray,
        patient_ids:       list[str] | None = None,
    ) -> None:
        assert len(image_features) == len(text_features) == len(clinical_features) == len(labels)

        self.image    = image_features.astype(np.float32)
        self.text     = text_features.astype(np.float32)
        self.clinical = clinical_features.astype(np.float32)
        self.labels   = labels.astype(np.int64)
        self.patient_ids = patient_ids or [str(i) for i in range(len(labels))]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image":      torch.tensor(self.image[idx]),
            "text":       torch.tensor(self.text[idx]),
            "clinical":   torch.tensor(self.clinical[idx]),
            "label":      torch.tensor(self.labels[idx]),
            "patient_id": self.patient_ids[idx],
        }
