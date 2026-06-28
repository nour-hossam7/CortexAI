"""
SegResNet model definition for BraTS2020 segmentation.

Selected over UNETR for Kaggle compatibility:
- Runs comfortably on P100/T4 (16GB) with batch_size=2 and ROI (128, 128, 128)
- 3-4x faster training than UNETR on same hardware
- Competitive Dice scores on BraTS (WT ~0.85+)

Author:
Mariam Mohamed
"""

import torch.nn as nn

from monai.networks.nets import SegResNet

from .config import Config

__all__ = [
    "build_model",
]


def build_model() -> nn.Module:
    """
    Build the SegResNet segmentation model.

    Architecture
    ------------
    - Encoder: residual blocks with progressive downsampling
    - Decoder: upsampling with skip connections
    - Output: raw logits of shape (B, OUT_CHANNELS, D, H, W)

    Input
    -----
    torch.Tensor : (B, 4, 128, 128, 128)
        4 MRI modalities — FLAIR, T1, T1ce, T2

    Output
    ------
    torch.Tensor : (B, 4, 128, 128, 128)
        Raw logits per class — {Background, NCR, Edema, ET}
        Apply softmax or argmax for final predictions.
    """

    model = SegResNet(

        spatial_dims=3,

        in_channels=Config.IN_CHANNELS,

        out_channels=Config.OUT_CHANNELS,

        init_filters=Config.INIT_FILTERS,

        blocks_down=Config.BLOCKS_DOWN,

        blocks_up=Config.BLOCKS_UP,

        dropout_prob=Config.DROPOUT_PROB,

    )

    return model
