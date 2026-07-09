"""
Student encoder for Knowledge Distillation.

Provides a lightweight transformer encoder (DistilBERT by default)
that can be trained via distillation to mimic a larger teacher
encoder (BioBERT / ClinicalBERT).  After training, the student can
replace the teacher at inference time with minimal quality loss.

The public API mirrors ``src/nlp_module/model.py`` so that callers
can treat teacher and student symmetrically.

Author:
Nour Hossam
"""

import logging
from typing import Tuple

import torch
from transformers import DistilBertModel, DistilBertTokenizer

from .distillation_config import DistillationConfig
from .model import mean_pooling, set_device

__all__ = [
    "build_student_encoder",
]

logger = logging.getLogger(__name__)


def build_student_encoder(
    model_name: str | None = None,
    device: torch.device | None = None,
    freeze: bool = False,
) -> Tuple[DistilBertTokenizer, DistilBertModel, torch.device]:
    """
    Load a lightweight student encoder suitable for knowledge distillation.

    Unlike :func:`src.nlp_module.model.build_encoder` this function
    returns the model in ``train()`` mode by default so that its
    parameters can be optimised during distillation.  Pass
    ``freeze=True`` to obtain a frozen eval-mode model for inference.

    Parameters
    ----------
    model_name : str | None
        HuggingFace model identifier.  Defaults to
        ``DistillationConfig.STUDENT_MODEL_NAME``.
    device : torch.device | None
        Target device.  Auto-selected when ``None``.
    freeze : bool
        When ``True`` all parameters are frozen and the model is set
        to ``eval()`` mode (inference-mode).  When ``False`` (default)
        the model is returned in ``train()`` mode.

    Returns
    -------
    tuple
        ``(tokenizer, model, device)``

    Raises
    ------
    OSError
        If the HuggingFace model cannot be downloaded or loaded.
    """
    if model_name is None:
        model_name = DistillationConfig.STUDENT_MODEL_NAME

    if device is None:
        device = set_device()

    logger.info("Loading student encoder: %s (device=%s)", model_name, device)

    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertModel.from_pretrained(model_name)
    except OSError as exc:
        logger.error("Failed to load student model '%s': %s", model_name, exc)
        raise

    model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    else:
        model.train()

    logger.info(
        "Student encoder loaded: %s parameters (trainable=%s)",
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    return tokenizer, model, device
