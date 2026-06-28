"""
Inference utilities for BraTS2020 SegResNet.

Provides:

- Model loading
- MRI segmentation
- Tumor volume estimation
- Image feature extraction

Author:
Mariam Mohamed
"""

from pathlib import Path

import torch

from monai.inferers import SlidingWindowInferer

from .config import Config
from .model import build_model

__all__ = [
    "load_model",
    "predict_mask",
    "calculate_tumor_volume",
    "extract_image_features",
]

def load_model(
    checkpoint_path: Path | None = None,
) -> torch.nn.Module:
    """
    Load trained SegResNet checkpoint.

    Handles two checkpoint formats, since both exist for this project:

    - Raw state_dict (what the training notebooks save, e.g. Notebook 08:
      `torch.save(model.state_dict(), "best_model.pth")`). This is the
      format of the actual best_model.pth currently on disk — the one
      all reported Dice scores and extracted features are based on.
    - Wrapped dict with a "model_state" key (what train.py's
      save_checkpoint() saves, alongside optimizer/scaler state for
      resuming training).

    Without this check, loading the existing best_model.pth through this
    function would raise KeyError: 'model_state'.
    """

    if checkpoint_path is None:
        checkpoint_path = (
            Config.CHECKPOINT_DIR /
            "best_model.pth"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        # Raw state_dict, as saved directly by the training notebooks.
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model

def predict_mask(
    model,
    sample,
    remap_to_brats: bool = False,
):
    """
    Predict segmentation mask
    for a single MRI volume.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded SegResNet, in eval() mode.
    sample : dict
        Must contain "image": Tensor (4, D, H, W).
    remap_to_brats : bool, default False
        If True, remaps internal label 3 (ET) back to the original
        BraTS convention label 4 before returning — matching the
        remap applied in Notebook 11 (save_prediction_nifti) when
        saving predictions as .nii.gz.
        If False (default), returns the internal contiguous labels
        {0: background, 1: NCR/NET, 2: ED, 3: ET} used throughout
        training, evaluation, and feature extraction.
        Downstream consumers (UI, fusion, Grad-CAM) should agree on
        which convention they expect — they are NOT interchangeable.

    Returns
    -------
    torch.Tensor
        Shape (D, H, W), integer class labels.
    """

    device = next(
        model.parameters()
    ).device

    image = sample["image"].unsqueeze(0).to(device)

    inferer = SlidingWindowInferer(

        roi_size=Config.ROI_SIZE,

        sw_batch_size=1,

        overlap=0.5,

    )

    with torch.no_grad():

        logits = inferer(
            image,
            model,
        )

        prediction = torch.argmax(
            logits,
            dim=1,
        )

    prediction = prediction.squeeze(0).cpu()

    if remap_to_brats:
        prediction = prediction.clone()
        prediction[prediction == 3] = 4

    return prediction

def calculate_tumor_volume(
    prediction,
):
    """
    Count tumor voxels.
    """

    tumor = prediction > 0

    return int(
        tumor.sum().item()
    )

def extract_image_features(
    model,
    sample,
):
    """
    Extract encoder bottleneck features for multimodal fusion.

    Registers a forward hook on the SegResNet encoder's last layer
    (`model.down_layers[-1]`) and global-average-pools its output over
    the spatial dimensions, producing a compact per-patient feature
    vector. This is the exact approach validated in Notebook 10
    (Deep Feature Extraction & Embedding Analysis), where it was run
    on all 56 validation patients and produced a (56, 256) feature
    matrix used for PCA / t-SNE / cosine-similarity analysis.

    Does not modify the model or its forward pass — the hook only
    reads the bottleneck activation and is removed before returning.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded SegResNet, in eval() mode, already on its target device.
    sample : dict
        Must contain "image": Tensor (4, D, H, W) — a preprocessed
        MRI volume (variable spatial size is fine; it is padded and
        center-cropped to ROI_SIZE internally).

    Returns
    -------
    np.ndarray
        Shape (256,) — bottleneck feature vector for this patient.
        Consumed downstream by the fusion module (Ammar) as the
        per-patient image feature representation.
    """

    import numpy as np

    from .preprocessing import pad_and_crop_128

    device = next(model.parameters()).device

    captured = {}

    def _hook(module, input, output):
        # output shape: (B, C, D, H, W) -> global average pool -> (B, C)
        captured["features"] = output.detach().mean(dim=[2, 3, 4])

    hook = model.down_layers[-1].register_forward_hook(_hook)

    image = sample["image"]
    patch = pad_and_crop_128(image).unsqueeze(0).to(device)  # (1, 4, 128, 128, 128)

    model.eval()

    with torch.no_grad():
        _ = model(patch)

    hook.remove()

    return captured["features"][0].cpu().numpy()  # (256,)