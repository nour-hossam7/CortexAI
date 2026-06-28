"""
Preprocessing pipeline for BraTS2020 MRI volumes.

This module provides reusable preprocessing transforms for:

- Training
- Validation
- Inference

Author:
Mariam Mohamed
"""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    MapLabelValued,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from monai.utils import set_determinism

from .config import Config

__all__ = [
    "set_seed",
    "get_base_transforms",
    "get_training_transforms",
    "get_patch_transforms",
    "get_validation_transforms",
    "get_inference_transforms",
    "pad_and_crop_128",
]


def set_seed() -> None:
    """
    Set deterministic behavior.
    """

    set_determinism(seed=Config.SEED)


def get_base_transforms() -> Compose:
    """
    Deterministic base transforms shared by training and validation.

    Order:
        Load → EnsureChannelFirst → EnsureType →
        MapLabel → CropForeground → ScaleIntensityRangePercentiles →
        NormalizeIntensity

    These are the only transforms cached by CacheDataset.
    Random transforms are never cached.
    """

    return Compose([

        LoadImaged(
            keys=["image", "label"]
        ),

        EnsureChannelFirstd(
            keys=["image", "label"]
        ),

        EnsureTyped(
            keys=["image", "label"]
        ),

        MapLabelValued(
            keys="label",
            orig_labels=Config.ORIGINAL_LABELS,
            target_labels=Config.TARGET_LABELS,
        ),

        CropForegroundd(
            keys=["image", "label"],
            source_key="label",
            margin=Config.CROP_MARGIN,
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


def get_training_transforms() -> Compose:
    """
    Full training pipeline.

    Extends base transforms with random patch sampling and augmentation.
    Applied inside CacheDataset — only deterministic stages are cached.

    SpatialPadd is required before RandCropByPosNegLabeld: after
    CropForegroundd, some patients end up smaller than ROI_SIZE in one
    or more dimensions (e.g. (46, 47, 52) observed during validation in
    Notebook 06). RandCropByPosNegLabeld cannot sample a 128³ patch from
    a smaller volume, so SpatialPadd pads up to ROI_SIZE first
    (no-op for volumes already >= ROI_SIZE in every dimension).

    Order:
        [Base Transforms]
            ↓
        SpatialPadd               ← pad to at least ROI_SIZE
            ↓
        RandCropByPosNegLabeld    ← patch sampling
            ↓
        RandFlipd (x3)            ← spatial augmentation
        RandRotate90d
            ↓
        RandGaussianNoised        ← intensity augmentation
        RandScaleIntensityd
        RandShiftIntensityd
    """

    return Compose([

        *get_base_transforms().transforms,

        SpatialPadd(
            keys=["image", "label"],
            spatial_size=Config.ROI_SIZE,
        ),

        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=Config.ROI_SIZE,
            pos=Config.POS_RATIO,
            neg=Config.NEG_RATIO,
            num_samples=Config.NUM_SAMPLES,
            image_key="image",
            image_threshold=0,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=0,
            prob=0.5,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=1,
            prob=0.5,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=2,
            prob=0.5,
        ),

        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=3,
        ),

        RandGaussianNoised(
            keys="image",
            prob=0.15,
            mean=0,
            std=0.01,
        ),

        RandScaleIntensityd(
            keys="image",
            factors=0.1,
            prob=0.5,
        ),

        RandShiftIntensityd(
            keys="image",
            offsets=0.1,
            prob=0.5,
        ),

    ])


def get_patch_transforms() -> Compose:
    """
    Patch sampling + augmentation only — for already-preprocessed .pt tensors.

    Use this (not get_training_transforms) when the input sample already
    contains loaded, cropped, and normalized tensors — i.e. when reading
    serialized .pt files produced by Notebook 06.5 via PreprocessedDataset.
    get_training_transforms() starts with LoadImaged, which expects file
    paths and will fail on tensors that are already in memory.

    This is the exact transform pipeline validated end-to-end in
    Notebook 08 (Advanced Training), applied directly on top of the
    serialized .pt tensors:

    Order:
        SpatialPadd               ← pad to at least ROI_SIZE
            ↓
        RandCropByPosNegLabeld    ← patch sampling
            ↓
        RandFlipd (x3)            ← spatial augmentation
        RandRotate90d
            ↓
        RandGaussianNoised        ← intensity augmentation
        RandScaleIntensityd
        RandShiftIntensityd

    Returns
    -------
    Compose
        Maps a single sample dict (image, label, ...) to a *list* of
        NUM_SAMPLES patch dicts (RandCropByPosNegLabeld always returns
        a list, even when num_samples == 1).
    """

    return Compose([

        SpatialPadd(
            keys=["image", "label"],
            spatial_size=Config.ROI_SIZE,
        ),

        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=Config.ROI_SIZE,
            pos=Config.POS_RATIO,
            neg=Config.NEG_RATIO,
            num_samples=Config.NUM_SAMPLES,
            image_key="image",
            image_threshold=0,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=0,
            prob=0.5,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=1,
            prob=0.5,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=2,
            prob=0.5,
        ),

        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=3,
        ),

        RandGaussianNoised(
            keys="image",
            prob=0.15,
            mean=0,
            std=0.01,
        ),

        RandScaleIntensityd(
            keys="image",
            factors=0.1,
            prob=0.5,
        ),

        RandShiftIntensityd(
            keys="image",
            offsets=0.1,
            prob=0.5,
        ),

    ])


def pad_and_crop_128(image):
    """
    Pad to at least 128 in each spatial dim, then center crop to exactly
    (128, 128, 128). Pure PyTorch — no MONAI transforms needed.

    Used by predict.py's extract_image_features() to bring a variable-size
    preprocessed volume (image, label-free) to the fixed input size the
    SegResNet encoder expects, without going through SlidingWindowInferer
    (single forward pass — bottleneck features only, no full-volume
    segmentation needed here).

    Validated in Notebook 10 (Deep Feature Extraction).

    Parameters
    ----------
    image : torch.Tensor  shape (4, D, H, W)

    Returns
    -------
    torch.Tensor  shape (4, 128, 128, 128)
    """

    import torch.nn.functional as F

    target = Config.ROI_SIZE[0]
    _, depth, height, width = image.shape

    pad_d = max(target - depth, 0)
    pad_h = max(target - height, 0)
    pad_w = max(target - width, 0)

    # F.pad pads last dims first: (W_left, W_right, H_left, H_right, D_left, D_right)
    image = F.pad(
        image,
        (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
        ),
        mode="constant",
        value=0,
    )

    _, depth2, height2, width2 = image.shape

    d0 = (depth2 - target) // 2
    h0 = (height2 - target) // 2
    w0 = (width2 - target) // 2

    return image[:, d0:d0 + target, h0:h0 + target, w0:w0 + target]


def get_validation_transforms() -> Compose:
    """
    Validation preprocessing — deterministic base transforms only.

    No patch sampling, no SpatialPadd. Full cropped volumes (variable
    size per patient) are passed to SlidingWindowInferer, which tiles
    the volume into overlapping windows at inference time. This is a
    deliberate design decision (re-confirmed in Notebook 06): padding
    or fixed-cropping validation volumes to ROI_SIZE was evaluated and
    rejected, since SlidingWindowInferer is the standard MONAI/BraTS
    approach for evaluating on full volumes and gives a more faithful
    Dice score than evaluating on a single fixed-size crop.
    """

    return get_base_transforms()


def get_inference_transforms() -> Compose:
    """
    Inference preprocessing pipeline.

    No label key — ground truth is unavailable at inference time.
    CropForegroundd uses source_key="image" (no mask available).

    Order:
        Load → Channel → Type → CropForeground →
        ScaleIntensityRangePercentiles → NormalizeIntensity

    ScaleIntensityRangePercentilesd must match training/validation
    exactly — applying a different intensity normalization at
    inference than at training time would shift the input
    distribution the model was trained on.
    """

    return Compose([

        LoadImaged(
            keys=["image"]
        ),

        EnsureChannelFirstd(
            keys=["image"]
        ),

        EnsureTyped(
            keys=["image"]
        ),

        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=Config.CROP_MARGIN,
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
