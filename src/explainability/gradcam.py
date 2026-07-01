"""
GradCAM for 3D SegResNet — BraTS2020

Produces a 3D activation heatmap showing which voxel regions drove
the SegResNet encoder's bottleneck representation — the same layer
used for feature extraction and fusion.

Two use cases are covered:

1. Segmentation GradCAM  — gradient of a specific output CLASS score
   w.r.t. down_layers[-1] activations (shows what drove tumor class N).

2. Feature GradCAM       — gradient of the global-average-pooled
   bottleneck feature norm w.r.t. down_layers[-1] activations
   (shows what drove the image features sent to the fusion module).

Author:
Ahmed Hossam
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from ..cv_module.config import Config
from ..cv_module.model import build_model
from ..cv_module.predict import load_model, predict_mask
from ..cv_module.preprocessing import pad_and_crop_128

__all__ = [
    "GradCAM3D",
    "compute_gradcam_segmentation",
    "overlay_gradcam_slice",
]


class GradCAM3D:
    """
    GradCAM for a 3D MONAI SegResNet.

    Hooks onto `model.down_layers[-1]` (output shape (B, 256, 16, 16, 16))
    and computes class-weighted activation maps upsampled back to the
    input volume size.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded SegResNet, in eval() mode, already on its target device.
    target_layer : torch.nn.Module | None
        Layer to hook. Defaults to model.down_layers[-1].
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer=None,
    ):
        self.model  = model
        self.device = next(model.parameters()).device
        self.layer  = target_layer if target_layer is not None else model.down_layers[-1]

        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        self._fwd_hook = self.layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.layer.register_full_backward_hook(self._save_gradient)

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def _save_activation(self, module, input, output):
        # NO .detach() — keep in computation graph so backward() reaches the hook
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Remove forward and backward hooks. Call after you are done."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── Core ──────────────────────────────────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        score_fn,
        output_size: tuple | None = None,
        relu: bool = True,
    ) -> np.ndarray:
        """
        Generate a GradCAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, 4, D, H, W) — batch of one, already on device.
        score_fn : callable
            Takes model output (logits or feature tensor) and returns a
            scalar to differentiate. This scalar defines what we are
            "explaining."
        output_size : tuple | None
            (D, H, W) to upsample the heatmap to. If None, returns at
            the bottleneck spatial resolution (16³ for default config).
        relu : bool
            If True, apply ReLU to the final heatmap — standard GradCAM
            convention that keeps only regions that positively contributed
            to the score. Set False to visualize both positive and negative
            contributions.

        Returns
        -------
        np.ndarray
            Shape (D, H, W) normalized to [0, 1].
        """

        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Scalar score to differentiate
        score = score_fn(output)

        # Backward
        self.model.zero_grad()
        score.backward()

        # Activations (B, C, D, H, W) and gradients (B, C, D, H, W)
        activations = self._activations   # (1, 256, 16, 16, 16)
        gradients   = self._gradients     # (1, 256, 16, 16, 16)

        # Global average pool gradients over spatial dims → weights (1, C)
        weights = gradients.mean(dim=[2, 3, 4], keepdim=True)  # (1, C, 1, 1, 1)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, D, H, W)

        if relu:
            cam = F.relu(cam)

        # Upsample to input size
        if output_size is not None:
            cam = F.interpolate(
                cam,
                size=output_size,
                mode="trilinear",
                align_corners=False,
            )

        cam = cam.detach().squeeze().cpu().numpy()   # .detach() required — cam still requires_grad after backward

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def compute_gradcam_segmentation(
    model: torch.nn.Module,
    sample: dict,
    target_class: int = 3,
) -> np.ndarray:
    """
    GradCAM explaining which voxels drove the prediction for a specific
    tumor sub-region (segmentation target class).

    Differentiates the mean logit score for `target_class` over all
    spatial positions w.r.t. the bottleneck activations.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded SegResNet, eval() mode, already on device.
    sample : dict
        Must contain "image": Tensor (4, D, H, W).
    target_class : int
        0=Background, 1=NCR/NET, 2=Edema, 3=ET.
        Default 3 (Enhancing Tumor) — typically most clinically relevant.

    Returns
    -------
    np.ndarray
        Shape (D, H, W) in [0,1] — heatmap at input volume size.
    """

    device = next(model.parameters()).device
    gradcam = GradCAM3D(model)

    image   = sample["image"]
    volume  = pad_and_crop_128(image).unsqueeze(0).to(device)
    D, H, W = image.shape[1], image.shape[2], image.shape[3]

    def score_fn(output):
        # output: (1, C, D, H, W) logits
        # Score = mean logit for target_class across all spatial positions
        return output[0, target_class].mean()

    heatmap = gradcam.generate(
        input_tensor=volume,
        score_fn=score_fn,
        output_size=(
            min(D, Config.ROI_SIZE[0]),
            min(H, Config.ROI_SIZE[1]),
            min(W, Config.ROI_SIZE[2]),
        ),
    )

    gradcam.remove_hooks()
    return heatmap


def overlay_gradcam_slice(
    image_slice: np.ndarray,
    heatmap_slice: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay a 2D GradCAM heatmap slice on top of an MRI image slice.

    Used for visualization in the notebook / UI — produces an RGB image
    where the heatmap is blended over the grayscale MRI.

    Parameters
    ----------
    image_slice : np.ndarray
        Shape (H, W) — one axial/coronal/sagittal slice of any MRI modality,
        normalized to [0, 1].
    heatmap_slice : np.ndarray
        Shape (H, W) — corresponding slice of the GradCAM heatmap, in [0, 1].
    alpha : float
        Heatmap opacity. Default 0.5.
    colormap : str
        Matplotlib colormap name for the heatmap. Default "jet".

    Returns
    -------
    np.ndarray
        Shape (H, W, 3) uint8 RGB image.
    """

    import matplotlib.cm as cm

    # Grayscale MRI → RGB
    img_norm = np.clip(image_slice, 0, 1)
    img_rgb  = np.stack([img_norm] * 3, axis=-1)

    # Heatmap → RGBA via colormap
    cmap   = cm.get_cmap(colormap)
    heat   = cmap(heatmap_slice)[:, :, :3]   # drop alpha channel

    # Blend
    blended = (1 - alpha) * img_rgb + alpha * heat
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return blended
