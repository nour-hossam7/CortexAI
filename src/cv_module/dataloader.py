"""
DataLoader utilities for BraTS2020.

Builds PyTorch DataLoaders for:

- Training
- Validation
- Testing

Author:
Mariam Mohamed
"""

from torch.utils.data import DataLoader

from monai.data import list_data_collate

from .config import Config

from .dataset import (
    get_train_dataset,
    get_validation_dataset,
    get_test_dataset,
)

__all__ = [
    "get_train_dataloader",
    "get_validation_dataloader",
    "get_test_dataloader",
]


def get_train_dataloader() -> DataLoader:
    """
    Build the training DataLoader.

    Each call to PreprocessedDataset.__getitem__ returns a *list* of
    NUM_SAMPLES patch dicts (RandCropByPosNegLabeld always returns a
    list, even for num_samples=1) — not a single dict. The default
    PyTorch collate_fn cannot batch a list of lists of dicts, so
    collate_fn=list_data_collate is required here. This matches the
    DataLoader configuration validated end-to-end in Notebook 08.

    Returns
    -------
    DataLoader
        Each batch: image (B * NUM_SAMPLES, 4, 128, 128, 128),
                    label (B * NUM_SAMPLES, 1, 128, 128, 128)
    """

    dataset = get_train_dataset()

    return DataLoader(
        dataset=dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=Config.SHUFFLE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.NUM_WORKERS > 0,
        collate_fn=list_data_collate,
    )


def get_validation_dataloader() -> DataLoader:
    """
    Build the validation DataLoader.

    Validation uses full cropped MRI volumes passed to SlidingWindowInferer.

    Returns
    -------
    DataLoader
        Each batch: image (1, 4, H, W, D), label (1, 1, H, W, D)
    """

    dataset = get_validation_dataset()

    return DataLoader(
        dataset=dataset,
        batch_size=Config.VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.NUM_WORKERS > 0,
    )


def get_test_dataloader() -> DataLoader:
    """
    Build the test DataLoader.

    Test data contains MRI volumes only —
    no ground-truth segmentation masks.

    Returns
    -------
    DataLoader
        Each batch: image (1, 4, H, W, D)
    """

    dataset = get_test_dataset()

    return DataLoader(
        dataset=dataset,
        batch_size=Config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.NUM_WORKERS > 0,
    )
