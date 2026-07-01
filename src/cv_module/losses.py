"""
Loss functions for BraTS2020 segmentation.

Author:
Mariam Mohamed
"""

import torch.nn as nn

from monai.losses import DiceCELoss

__all__ = [
    "build_loss",
]

def build_loss() -> nn.Module:
    """
    Build the segmentation loss.

    Returns
    -------
    DiceCELoss

    Combines:

    - Dice Loss
    - Cross Entropy Loss

    Recommended by MONAI for multi-class
    brain tumor segmentation.

    Deliberately matches DiceCELoss(to_onehot_y=True, softmax=True)
    exactly as used in Notebook 08 (Advanced Training) — the run that
    actually produced the current best_model.pth. squared_pred,
    smooth_nr, and smooth_dr are left at MONAI's defaults rather than
    Config.SQUARED_PRED / Config.SMOOTH_NR / Config.SMOOTH_DR, since
    Notebook 08 never set those and resuming training with a different
    loss shape than the one the checkpoint was trained under would
    silently change the optimization target.
    """

    loss = DiceCELoss(

        to_onehot_y=True,

        softmax=True,

    )

    return loss