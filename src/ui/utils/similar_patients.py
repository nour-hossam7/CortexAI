"""Similar patient retrieval with fallbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.ui.utils.paths import get_paths


def find_similar_patients_for_case(
    patient_id: str,
    unified_repr: np.ndarray | None,
    image_features: np.ndarray | None,
    stats: Any | None,
    fusion: Any | None,
    top_k: int = 5,
) -> pd.DataFrame:
    cohort = _load_cohort_representations()
    if cohort is None and image_features is not None:
        cohort = _load_image_feature_cohort()

    if cohort is None or cohort["matrix"].shape[0] < 2:
        return pd.DataFrame(columns=["rank", "patient_id", "similarity", "wt_volume_cm3", "risk_label", "confidence"])

    matrix = cohort["matrix"]
    ids = cohort["patient_ids"]
    meta = cohort["meta"]

    if unified_repr is not None:
        query = unified_repr.reshape(1, -1)
    elif image_features is not None:
        query = np.asarray(image_features, dtype=np.float32).reshape(1, -1)
        if matrix.shape[1] != query.shape[1]:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    sims = cosine_similarity(query, matrix)[0]
    order = np.argsort(sims)[::-1]

    rows = []
    rank = 1
    for idx in order:
        pid = ids[idx]
        if pid == patient_id:
            continue
        row_meta = meta[meta["patient_id"] == pid]
        wt = float(row_meta["wt_volume"].iloc[0]) if len(row_meta) and "wt_volume" in row_meta else np.nan
        risk = row_meta["risk_label"].iloc[0] if len(row_meta) and "risk_label" in row_meta else "—"
        conf = row_meta["confidence"].iloc[0] if len(row_meta) and "confidence" in row_meta else np.nan
        rows.append({
            "rank": rank,
            "patient_id": pid,
            "similarity": round(float(sims[idx]), 4),
            "wt_volume_cm3": round(wt * _voxel_to_cm3_factor(stats), 3) if not np.isnan(wt) else np.nan,
            "risk_label": _risk_name(risk),
            "confidence": round(float(conf), 3) if not np.isnan(conf) else np.nan,
        })
        rank += 1
        if rank > top_k:
            break
    return pd.DataFrame(rows)


def _load_cohort_representations() -> dict | None:
    paths = get_paths()
    repr_dir = paths.fusion_repr
    if not repr_dir.exists():
        return None

    parts, ids, meta_frames = [], [], []
    for split in ("train", "validation", "test"):
        npy = repr_dir / f"{split}_unified_repr.npy"
        csv = repr_dir / f"{split}_repr_metadata.csv"
        if npy.exists() and csv.exists():
            parts.append(np.load(npy))
            m = pd.read_csv(csv)
            ids.extend(m["patient_id"].tolist())
            meta_frames.append(m)

    if not parts:
        return None

    matrix = np.vstack(parts)
    meta = pd.concat(meta_frames, ignore_index=True)
    return {"matrix": matrix, "patient_ids": ids, "meta": meta}


def _load_image_feature_cohort() -> dict | None:
    paths = get_paths()
    result_dir = paths.cv_results
    parts, ids, meta_frames = [], [], []

    for split in ("train", "validation", "test"):
        npy = result_dir / f"bottleneck_features_{split}.npy"
        csv = result_dir / f"{split}_metadata.csv"
        if not npy.exists() and split == "validation":
            npy = result_dir / "bottleneck_features.npy"
            csv = result_dir / "feature_metadata.csv"
        if npy.exists() and csv.exists():
            parts.append(np.load(npy))
            m = pd.read_csv(csv)
            ids.extend(m["patient_id"].tolist())
            meta_frames.append(m)

    if not parts:
        return None
    return {"matrix": np.vstack(parts), "patient_ids": ids, "meta": pd.concat(meta_frames, ignore_index=True)}


def _voxel_to_cm3_factor(stats) -> float:
    if stats is None:
        return 1e-3
    sx, sy, sz = stats.spacing_mm
    return sx * sy * sz / 1000.0


def _risk_name(val) -> str:
    mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    if isinstance(val, (int, np.integer)):
        return mapping.get(int(val), str(val))
    return str(val)
