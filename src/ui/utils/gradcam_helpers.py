"""Grad-CAM helpers for WT / TC / ET target regions."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.cv_module.config import Config
from src.explainability.gradcam import GradCAM3D
from src.cv_module.preprocessing import pad_and_crop_128

TARGET_MAP = {
    "ET": 3,
    "Edema": 2,
    "NCR": 1,
}


def compute_gradcam_for_target(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: str = "ET",
    relu: bool = True,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for ET, TC (NCR+ET), or WT (all tumor classes).
    """
    device = next(model.parameters()).device
    gradcam = GradCAM3D(model)
    volume = pad_and_crop_128(image).unsqueeze(0).to(device)
    d, h, w = image.shape[1], image.shape[2], image.shape[3]
    output_size = (min(d, Config.ROI_SIZE[0]), min(h, Config.ROI_SIZE[1]), min(w, Config.ROI_SIZE[2]))

    score_fn = _score_fn_for_target(target)

    heatmap = gradcam.generate(
        input_tensor=volume,
        score_fn=score_fn,
        output_size=output_size,
        relu=relu,
    )
    gradcam.remove_hooks()

    if heatmap.shape != (d, h, w):
        heatmap = _resize_volume(heatmap, (d, h, w))
    return heatmap


def apply_heatmap_intensity(heatmap: np.ndarray, gain: float = 1.0) -> np.ndarray:
    hm = np.clip(heatmap * gain, 0, None)
    mx = hm.max()
    if mx > 1e-8:
        hm = hm / mx
    return np.clip(hm, 0, 1)


def overlay_heatmap_on_slice(
    gray: np.ndarray,
    heatmap_slice: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "jet",
) -> np.ndarray:
    import matplotlib.cm as cm

    gray = np.clip(gray, 0, 1)
    base = np.stack([gray] * 3, axis=-1)
    cmap = cm.get_cmap(cmap_name)
    heat_rgb = cmap(heatmap_slice)[:, :, :3]
    mask = heatmap_slice > 0.05
    out = base.copy()
    out[mask] = (1 - alpha) * base[mask] + alpha * heat_rgb[mask]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _score_fn_for_target(target: str) -> Callable:
    target = target.upper()

    def score_fn(output: torch.Tensor) -> torch.Tensor:
        logits = output[0]
        if target == "WT":
            return logits[[1, 2, 3]].mean()
        if target == "TC":
            return logits[[1, 3]].mean()
        if target == "ET":
            return logits[3].mean()
        cls = TARGET_MAP.get(target, 3)
        return logits[cls].mean()

    return score_fn


def _resize_volume(vol: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=size, mode="trilinear", align_corners=False)
    return t.squeeze().numpy()
