"""
Frozen BERT-family encoder wrapper for radiology report embedding.

Provides the tokenizer + model loading and the attention-mask-aware
mean pooling used to turn a variable-length report into a single
fixed-size feature vector.

Validated end-to-end in Notebooks 06a (BioBERT) and 06b (ClinicalBERT)
— same pooling logic, same normalization, applied identically to both
candidate encoders so they can be swapped via Config.MODEL_NAME.

Author:
Mariam Mohamed
"""

from typing import Tuple

import torch
from transformers import BertModel, BertTokenizer

from .config import Config

__all__ = [
    "set_device",
    "build_encoder",
    "mean_pooling",
]


def set_device() -> torch.device:
    """
    Select device for the encoder. Mirrors cv_module.train.set_device().
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_encoder(
    model_name: str | None = None,
) -> Tuple[BertTokenizer, BertModel, torch.device]:
    """
    Load a frozen pretrained encoder and its tokenizer.

    The model is set to eval() mode and moved to the selected device.
    It is never fine-tuned anywhere in this module — only used to
    extract fixed embeddings (Notebooks 06a/06b) — so no optimizer or
    gradient tracking is needed here.

    Parameters
    ----------
    model_name : str | None
        HuggingFace model identifier. Defaults to Config.MODEL_NAME.
        Must be one of Config.AVAILABLE_MODELS to keep embeddings
        comparable to what was validated in Notebooks 05a/05b/06a/06b/07.

    Returns
    -------
    tuple
        (tokenizer, model, device)
    """

    if model_name is None:
        model_name = Config.MODEL_NAME

    if model_name not in Config.AVAILABLE_MODELS:
        raise ValueError(
            f"'{model_name}' was not one of the models validated in "
            f"Notebooks 05a/05b/06a/06b/07: {Config.AVAILABLE_MODELS}. "
            f"Using an unvalidated encoder here would produce embeddings "
            f"with unknown statistics relative to the saved .npy files."
        )

    device = set_device()

    # Both validated encoders are BERT-family models, so we can load the
    # concrete slow tokenizer class directly and avoid Hugging Face's
    # backend-tokenizer conversion path.
    tokenizer = BertTokenizer.from_pretrained(model_name)

    model = BertModel.from_pretrained(model_name)

    model.to(device)


    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return tokenizer, model, device


def mean_pooling(
    model_output,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Attention-mask-aware mean pooling over token embeddings.

    Averages last_hidden_state only over real (non-padding) token
    positions — naive mean pooling over all MAX_LENGTH positions would
    be biased by [PAD] token embeddings, which are NOT zero vectors for
    BERT-family models (confirmed in Notebook 01 §11: per-token L2 norms
    at padding positions are non-zero). This resolves the pooling
    decision originally deferred to Notebooks 05/06.

    Parameters
    ----------
    model_output : transformers.modeling_outputs.BaseModelOutput
        Output of the encoder forward pass; uses .last_hidden_state,
        shape (B, L, H).
    attention_mask : torch.Tensor
        Shape (B, L), 1 for real tokens, 0 for padding.

    Returns
    -------
    torch.Tensor
        Shape (B, H) — mean-pooled embedding per report.
    """

    embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    summed = torch.sum(embeddings * mask, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed / counted
