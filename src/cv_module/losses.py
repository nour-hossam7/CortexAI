"""
Loss functions for BraTS2020 segmentation.

Author:
Mariam Mohamed
"""

import torch.nn as nn

from .config import Config

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
    """

    loss = DiceCELoss(

        to_onehot_y=True,

        softmax=True,

        squared_pred=Config.SQUARED_PRED,

        smooth_nr=Config.SMOOTH_NR,

        smooth_dr=Config.SMOOTH_DR,

    )

    return loss