"""Load MRI volumes from .pt, .nii, and .nii.gz uploads."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
)

from src.cv_module.config import Config as CVConfig

# MONAI prints "applying transform LoadImaged…" on every compose step — noisy in Streamlit.
logging.getLogger("monai").setLevel(logging.WARNING)

MODALITY_INDEX = {m: i for i, m in enumerate(CVConfig.MODALITIES)}


def get_volume_inference_transforms() -> Compose:
    """
    Inference transforms for in-memory volumes already stacked as (4, D, H, W).

    EnsureChannelFirstd is omitted — it fails on raw 4D numpy without MONAI
    metadata (the red "applying transform EnsureChannelFirstd" error in the UI).
    """
    return Compose([
        EnsureTyped(keys=["image"]),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=CVConfig.CROP_MARGIN,
        ),
        ScaleIntensityRangePercentilesd(
            keys="image",
            lower=1,
            upper=99,
            b_min=0,
            b_max=1,
            clip=True,
            channel_wise=True,
        ),
        NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=True,
        ),
    ])


def format_transform_error(exc: BaseException) -> str:
    """Unwrap MONAI Compose errors — str(exc) is often only 'applying transform …'."""
    chain: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        text = str(current).strip()
        if text and text not in chain:
            chain.append(text)
        current = current.__cause__
    if len(chain) > 1:
        return chain[0] + "\n\nCause: " + chain[-1]
    return chain[0] if chain else repr(exc)


def load_pt_volume(path: Path | str) -> dict[str, Any]:
    sample = torch.load(path, map_location="cpu", weights_only=False)
    image = sample["image"]
    if isinstance(image, torch.Tensor):
        image = image.float()
    metadata = sample.get("metadata", {})
    patient_id = sample.get("patient_id", Path(path).stem.replace(".pt", ""))
    spacing = _spacing_from_metadata(metadata)
    return {
        "image": image,
        "patient_id": patient_id,
        "spacing_mm": spacing,
        "affine": metadata.get("affine"),
        "source": str(path),
        "modalities": list(CVConfig.MODALITIES),
    }


def load_nifti_modalities(
    files: dict[str, Path | str],
    patient_id: str = "Uploaded_Patient",
) -> dict[str, Any]:
    """
    Load four BraTS modalities from disk and apply the training-matched
    inference transforms without MONAI LoadImaged (faster + reliable for
    Streamlit temp uploads on Windows).
    """
    ordered_paths = [Path(files[m]) for m in CVConfig.MODALITIES if m in files]
    if len(ordered_paths) != len(CVConfig.MODALITIES):
        raise ValueError(
            f"Provide all four modalities: {CVConfig.MODALITIES}. "
            f"Received: {list(files.keys())}"
        )

    for path in ordered_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Modality file not found: {path}")

    volumes = [_load_nifti_array(path) for path in ordered_paths]
    shapes = {v.shape for v in volumes}
    if len(shapes) != 1:
        raise ValueError(
            "All four modalities must share the same shape. "
            f"Got: {dict(zip(CVConfig.MODALITIES, [v.shape for v in volumes]))}"
        )

    stacked = np.stack(volumes, axis=0).astype(np.float32)
    # Channel-first (4, X, Y, Z) — same layout as LoadImaged + EnsureChannelFirstd in training.
    image = MetaTensor(
        stacked,
        meta={
            "original_channel_dim": 0,
            "spatial_shape": stacked.shape[1:],
        },
    )
    processed = get_volume_inference_transforms()({"image": image})
    tensor = _to_float_tensor(processed["image"])
    spacing = _spacing_from_nifti(ordered_paths[0])

    return {
        "image": tensor,
        "patient_id": patient_id,
        "spacing_mm": spacing,
        "affine": None,
        "source": "nifti_upload",
        "modalities": list(CVConfig.MODALITIES),
        "paths": [str(p) for p in ordered_paths],
    }


def preprocess_paths(image_paths: list[str] | dict[str, str]) -> dict[str, Any]:
    if isinstance(image_paths, dict):
        return load_nifti_modalities(image_paths)
    files = {mod: Path(image_paths[i]) for i, mod in enumerate(CVConfig.MODALITIES)}
    return load_nifti_modalities(files)


def save_upload_to_temp(uploaded_file, suffix: str | None = None) -> Path:
    """Persist a Streamlit upload to a temp path, keeping the original extension."""
    original = Path(uploaded_file.name)
    ext = suffix or "".join(original.suffixes) or ".bin"
    if not ext.startswith("."):
        ext = f".{ext}"
    tmp_dir = Path(tempfile.mkdtemp(prefix="cortexai_"))
    path = tmp_dir / f"{original.stem}{ext}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def store_upload_paths(upload_key: str, files: dict[str, Path]) -> None:
    import streamlit as st

    st.session_state[upload_key] = {k: str(v) for k, v in files.items()}


def get_stored_upload_paths(upload_key: str) -> dict[str, Path] | None:
    import streamlit as st

    raw = st.session_state.get(upload_key)
    if not raw or len(raw) != len(CVConfig.MODALITIES):
        return None
    return {k: Path(v) for k, v in raw.items()}


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().numpy()


def get_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def normalize_slice(sl: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    sl = sl.astype(np.float32)
    lo = float(np.percentile(sl, 1)) if lo is None else lo
    hi = float(np.percentile(sl, 99)) if hi is None else hi
    if hi - lo < 1e-8:
        return np.zeros_like(sl)
    out = (sl - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def _load_nifti_array(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required to load NIfTI uploads.") from exc

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume in {path.name}, got shape {data.shape}")
    return data


def _to_float_tensor(value) -> torch.Tensor:
    if hasattr(value, "as_tensor"):
        return value.as_tensor().float()
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _spacing_from_metadata(metadata: dict) -> tuple[float, float, float]:
    spacing = metadata.get("spacing") or metadata.get("pixdim")
    if spacing is not None and len(spacing) >= 3:
        vals = spacing[-3:] if len(spacing) > 3 else spacing[:3]
        return tuple(float(v) for v in vals)
    return (1.0, 1.0, 1.0)


def _spacing_from_nifti(path: str | Path) -> tuple[float, float, float]:
    try:
        import nibabel as nib

        img = nib.load(str(path))
        zooms = img.header.get_zooms()[:3]
        return tuple(float(z) for z in zooms)
    except Exception:
        return (1.0, 1.0, 1.0)
