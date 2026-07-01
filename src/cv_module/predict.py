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
from typing import Dict, List

import torch

from monai.inferers import SlidingWindowInferer

from .config import Config
from .model import build_model
from .preprocessing import get_inference_transforms

__all__ = [
    "load_model",
    "build_inference_sample",
    "predict_mri",
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

def build_inference_sample(
    image_paths: List[str] | Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """
    Build a preprocessed sample dict from raw MRI files for a brand-new
    patient — i.e. one that was never run through Notebook 06.5 and has
    no serialized .pt file in datasets/processed/cv/test/.

    This is the missing link for predict_mri(): it applies
    get_inference_transforms() (preprocessing.py), which uses
    CropForegroundd(source_key="image") — the no-label-available variant,
    matching real deployment conditions exactly. Do NOT substitute
    get_base_transforms() here; that version crops with source_key="label"
    and requires a ground-truth mask that a new patient will not have.

    Parameters
    ----------
    image_paths : list[str] | dict[str, str]
        Either a list of 4 file paths in the channel order contract
        from Notebook 04 — [FLAIR, T1, T1ce, T2] — or a dict keyed by
        modality name (any subset/order of Config.MODALITIES), e.g.
        {"flair": "...", "t1": "...", "t1ce": "...", "t2": "..."}.

    Returns
    -------
    dict
        {"image": Tensor (4, D, H, W) float32}
        Already cropped, intensity-scaled, and normalized — ready for
        predict_mask() / extract_image_features().
    """

    if isinstance(image_paths, dict):
        ordered_paths = [
            str(image_paths[modality]) for modality in Config.MODALITIES
        ]
    else:
        ordered_paths = [str(path) for path in image_paths]

        if len(ordered_paths) != len(Config.MODALITIES):
            raise ValueError(
                f"Expected {len(Config.MODALITIES)} modality paths "
                f"({Config.MODALITIES}), got {len(ordered_paths)}."
            )

    transform = get_inference_transforms()

    processed = transform({"image": ordered_paths})

    return {"image": processed["image"].as_tensor()}


def predict_mri(
    model,
    image_paths: List[str] | Dict[str, str],
    remap_to_brats: bool = False,
):
    """
    End-to-end prediction for a single new MRI case — the function
    referenced in the project task list as predict_mri(image_path).

    Loads and preprocesses raw NIfTI files (no precomputed .pt file
    required), then runs sliding-window inference exactly like
    predict_mask().

    Parameters
    ----------
    model : torch.nn.Module
        Loaded SegResNet, in eval() mode (see load_model()).
    image_paths : list[str] | dict[str, str]
        Raw modality file paths — see build_inference_sample().
    remap_to_brats : bool, default False
        See predict_mask().

    Returns
    -------
    torch.Tensor
        Shape (D, H, W), integer class labels.
    """

    sample = build_inference_sample(image_paths)

    return predict_mask(
        model=model,
        sample=sample,
        remap_to_brats=remap_to_brats,
    )


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
    image,
    device,
):
    """
    Extract encoder bottleneck features for multimodal fusion.

    Registers a forward hook on the SegResNet encoder's last layer
    (`model.down_layers[-1]`) and global-average-pools its output over
    the spatial dimensions, producing a compact per-patient feature
    vector.

    Signature matches Notebook 10 (Deep Feature Extraction & Embedding
    Analysis) exactly — `(model, image, device)`, where `image` already
    carries a batch dimension — so code can be copied between this file
    and the notebook in either direction without a TypeError.

    In the notebook the typical call pattern is:
        image = sample["image"].unsqueeze(0)   # (1, 4, D, H, W)
        feat  = extract_image_features(model, image, device)

    Does not modify the model or its forward pass — the hook only
    reads the bottleneck activation and is removed before returning.

    Parameters
    ----------
    model : torch.nn.Module
        SegResNet, in eval() mode, already on `device`.
    image : torch.Tensor
        Shape (1, 4, D, H, W) — preprocessed MRI volume with the batch
        dimension already added (variable spatial size is fine; padded
        and center-cropped to ROI_SIZE internally via pad_and_crop_128).
    device : torch.device
        Where to run the forward pass. Passed explicitly rather than
        inferred from model.parameters(), matching Notebook 10 exactly.

    Returns
    -------
    np.ndarray
        Shape (256,) — bottleneck feature vector for this patient.
        Consumed downstream by the fusion module (Ammar) as the
        per-patient image feature representation.

    See Also
    --------
    dataset.load_image_features : load ALL precomputed bottleneck
        features (saved by Notebook 10) for a full split, indexed by
        patient_id — the fast path for the fusion module.
    """

    import numpy as np

    from .preprocessing import pad_and_crop_128

    captured = {}

    def _hook(module, input, output):
        # output shape: (B, C, D, H, W) -> global average pool -> (B, C)
        captured["features"] = output.detach().mean(dim=[2, 3, 4])

    hook = model.down_layers[-1].register_forward_hook(_hook)

    patch = pad_and_crop_128(image[0]).unsqueeze(0).to(device)  # (1, 4, 128, 128, 128)

    model.eval()

    with torch.no_grad():
        _ = model(patch)

    hook.remove()

    return captured["features"][0].cpu().numpy()  # (256,)