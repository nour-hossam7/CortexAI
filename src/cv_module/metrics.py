"""
Evaluation metrics for BraTS2020 segmentation.

Provides Dice metric and post-processing utilities.

Author:
Mariam Mohamed
"""

from monai.metrics import DiceMetric

from .config import Config

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

__all__ = [
    "build_metric",
    "get_post_transforms",
]

def build_metric() -> DiceMetric:
    """
    Build Dice metric for multi-class segmentation.

    Returns
    -------
    DiceMetric
        Mean Dice score across tumor classes.
    """

    return DiceMetric(

        include_background=Config.INCLUDE_BACKGROUND,

        reduction=Config.METRIC_REDUCTION,

        get_not_nans=False,

    )

def get_post_transforms():
    """
    Build post-processing transforms.

    Returns
    -------
    tuple
        (post_prediction, post_label)
    """

    post_prediction = Compose([

        Activations(softmax=True),

        AsDiscrete(
            argmax=True,
            to_onehot=Config.OUT_CHANNELS,
        ),

    ])

    post_label = Compose([

        AsDiscrete(
            to_onehot=Config.OUT_CHANNELS,
        ),

    ])

    return post_prediction, post_label