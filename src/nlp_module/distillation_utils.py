"""
Utility functions for Knowledge Distillation of the NLP encoder.

Provides:

* Loss functions (MSE, Cosine, MSE + Cosine hybrid).
* Embedding-similarity metrics.
* Model comparison helpers (latency, parameter count, size).
* A lightweight ``TextDataset`` for creating PyTorch DataLoaders
  from a list of report texts.

All functions accept optional ``device`` parameters and log their
results through the standard ``logging`` module.

Author:
Nour Hossam
"""

import logging
import time
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import Config as NLPConfig
from .distillation_config import DistillationConfig
from .model import mean_pooling

__all__ = [
    "TextDataset",
    "mse_loss",
    "cosine_embedding_loss",
    "mse_cosine_loss",
    "get_loss_fn",
    "compute_cosine_similarity",
    "compute_mse",
    "count_parameters",
    "measure_model_size_mb",
    "measure_latency",
    "compare_models",
]

logger = logging.getLogger(__name__)


# ===================================================================
# Dataset
# ===================================================================


class TextDataset(Dataset):
    """
    In-memory dataset of report texts for distillation training.

    Parameters
    ----------
    texts : list of str
        Cleaned report texts.
    """

    def __init__(self, texts: List[str]) -> None:
        if not texts:
            raise ValueError("texts list must not be empty")

        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


# ===================================================================
# Loss functions
# ===================================================================


def mse_loss(
    teacher_embedding: torch.Tensor,
    student_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Mean Squared Error between teacher and student embeddings.

    Both tensors should be shape ``(batch, embedding_dim)``.

    Parameters
    ----------
    teacher_embedding : torch.Tensor
        Embeddings from the frozen teacher model.
    student_embedding : torch.Tensor
        Embeddings from the student model (requires gradients).

    Returns
    -------
    torch.Tensor
        Scalar MSE loss.
    """
    return F.mse_loss(student_embedding, teacher_embedding.detach())


def cosine_embedding_loss(
    teacher_embedding: torch.Tensor,
    student_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    1 - mean cosine similarity between teacher and student embeddings.

    Parameters
    ----------
    teacher_embedding : torch.Tensor
        Shape ``(batch, embedding_dim)``.
    student_embedding : torch.Tensor
        Shape ``(batch, embedding_dim)``.

    Returns
    -------
    torch.Tensor
        Scalar cosine loss.
    """
    teacher_norm = F.normalize(teacher_embedding.detach(), p=2, dim=1)
    student_norm = F.normalize(student_embedding, p=2, dim=1)
    return 1.0 - (teacher_norm * student_norm).sum(dim=1).mean()


def mse_cosine_loss(
    teacher_embedding: torch.Tensor,
    student_embedding: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Hybrid loss combining MSE and cosine embedding loss.

    ``loss = alpha * mse + (1 - alpha) * cosine``

    Parameters
    ----------
    teacher_embedding : torch.Tensor
        Shape ``(batch, embedding_dim)``.
    student_embedding : torch.Tensor
        Shape ``(batch, embedding_dim)``.
    alpha : float
        Weight for the MSE term (default 0.5).

    Returns
    -------
    torch.Tensor
        Scalar hybrid loss.
    """
    mse = mse_loss(teacher_embedding, student_embedding)
    cosine = cosine_embedding_loss(teacher_embedding, student_embedding)
    return alpha * mse + (1.0 - alpha) * cosine


def get_loss_fn(loss_type: str | None = None) -> Callable:
    """
    Factory that returns the loss function matching *loss_type*.

    Parameters
    ----------
    loss_type : str | None
        One of ``"mse"``, ``"cosine"``, ``"mse_cosine"``.
        Defaults to ``DistillationConfig.LOSS_TYPE``.

    Returns
    -------
    callable
        Loss function with signature
        ``(teacher_emb, student_emb) -> scalar_tensor``.

    Raises
    ------
    ValueError
        If *loss_type* is unknown.
    """
    if loss_type is None:
        loss_type = DistillationConfig.LOSS_TYPE

    registry = {
        "mse": mse_loss,
        "cosine": cosine_embedding_loss,
        "mse_cosine": mse_cosine_loss,
    }

    if loss_type not in registry:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            f"Must be one of {list(registry.keys())}."
        )

    return registry[loss_type]


# ===================================================================
# Evaluation metrics
# ===================================================================


@torch.no_grad()
def compute_cosine_similarity(
    teacher_embeddings: torch.Tensor,
    student_embeddings: torch.Tensor,
) -> float:
    """
    Mean cosine similarity between teacher and student embedding sets.

    Parameters
    ----------
    teacher_embeddings : torch.Tensor
        Shape ``(N, embedding_dim)``.
    student_embeddings : torch.Tensor
        Shape ``(N, embedding_dim)``.

    Returns
    -------
    float
        Mean cosine similarity.
    """
    teacher_norm = F.normalize(teacher_embeddings, p=2, dim=1)
    student_norm = F.normalize(student_embeddings, p=2, dim=1)
    sim = (teacher_norm * student_norm).sum(dim=1)
    return sim.mean().item()


@torch.no_grad()
def compute_mse(
    teacher_embeddings: torch.Tensor,
    student_embeddings: torch.Tensor,
) -> float:
    """
    Mean Squared Error between teacher and student embedding sets.

    Parameters
    ----------
    teacher_embeddings : torch.Tensor
        Shape ``(N, embedding_dim)``.
    student_embeddings : torch.Tensor
        Shape ``(N, embedding_dim)``.

    Returns
    -------
    float
        MSE value.
    """
    return F.mse_loss(student_embeddings, teacher_embeddings).item()


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters of a model.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    tuple
        ``(total_params, trainable_params)``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_model_size_mb(model: torch.nn.Module) -> float:
    """
    Approximate memory footprint of model parameters in megabytes.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    float
        Size in MB (based on parameter count and dtype).
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2)


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    num_runs: int = 50,
    warmup_runs: int = 10,
) -> Tuple[float, float, float]:
    """
    Measure inference latency of an encoder on a list of texts.

    Parameters
    ----------
    model : torch.nn.Module
        Encoder model (teacher or student), in ``eval()`` mode.
    tokenizer
        Matching tokenizer.
    texts : list of str
        Report texts to encode.
    device : torch.device
        Target device.
    num_runs : int
        Number of timed runs (default 50).
    warmup_runs : int
        Number of untimed warm-up runs (default 10).

    Returns
    -------
    tuple
        ``(mean_ms, std_ms, p99_ms)`` inference time in milliseconds.
    """
    model.eval()
    latencies: List[float] = []

    encoded = tokenizer(
        texts,
        max_length=NLPConfig.MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    for _ in range(warmup_runs + num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = model(**encoded)
        pooled = mean_pooling(outputs, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if _ >= warmup_runs:
            latencies.append(elapsed_ms)

    latencies.sort()
    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    p99_ms = float(latencies[int(len(latencies) * 0.99)])

    return mean_ms, std_ms, p99_ms


def compare_models(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    teacher_tokenizer,
    student_tokenizer,
    texts: List[str],
    teacher_embeddings: torch.Tensor | None = None,
    student_embeddings: torch.Tensor | None = None,
    device: torch.device | None = None,
    num_latency_runs: int = 50,
) -> dict:
    """
    Generate a side-by-side comparison dictionary between teacher and
    student models.

    Parameters
    ----------
    teacher_model : torch.nn.Module
    student_model : torch.nn.Module
    teacher_tokenizer
    student_tokenizer
    texts : list of str
        Texts to use for latency and (optionally) embedding metrics.
    teacher_embeddings : torch.Tensor or None
        Precomputed teacher embeddings.  Computed on-the-fly if
        ``None`` and ``texts`` is provided.
    student_embeddings : torch.Tensor or None
        Precomputed student embeddings.  Computed on-the-fly if
        ``None`` and ``texts`` is provided.
    device : torch.device or None
        Auto-detected if ``None``.
    num_latency_runs : int
        Latency measurement runs.

    Returns
    -------
    dict
        Nested dictionary ``{"Teacher": {key: val}, "Student": {key: val}}``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_total, teacher_trainable = count_parameters(teacher_model)
    student_total, student_trainable = count_parameters(student_model)

    result = {
        "Teacher": {
            "model_name": str(type(teacher_model).__name__),
            "total_params": teacher_total,
            "trainable_params": teacher_trainable,
            "size_mb": round(measure_model_size_mb(teacher_model), 1),
        },
        "Student": {
            "model_name": str(type(student_model).__name__),
            "total_params": student_total,
            "trainable_params": student_trainable,
            "size_mb": round(measure_model_size_mb(student_model), 1),
        },
    }

    result["Reduction"] = {
        "param_reduction_pct": round(
            (1 - student_total / teacher_total) * 100, 1
        ),
        "size_reduction_pct": round(
            (1 - result["Student"]["size_mb"] / result["Teacher"]["size_mb"])
            * 100,
            1,
        ),
    }

    if texts:
        logger.info("Measuring inference latency (%d runs)...", num_latency_runs)
        t_mean, t_std, t_p99 = measure_latency(
            teacher_model, teacher_tokenizer, texts, device,
            num_runs=num_latency_runs,
        )
        s_mean, s_std, s_p99 = measure_latency(
            student_model, student_tokenizer, texts, device,
            num_runs=num_latency_runs,
        )
        result["Teacher"]["latency_mean_ms"] = round(t_mean, 2)
        result["Teacher"]["latency_std_ms"] = round(t_std, 2)
        result["Teacher"]["latency_p99_ms"] = round(t_p99, 2)
        result["Student"]["latency_mean_ms"] = round(s_mean, 2)
        result["Student"]["latency_std_ms"] = round(s_std, 2)
        result["Student"]["latency_p99_ms"] = round(s_p99, 2)
        result["Speedup"] = {
            "mean_speedup_x": round(t_mean / s_mean, 2) if s_mean > 0 else float("inf"),
        }

    if teacher_embeddings is None and student_embeddings is None and texts:
        logger.info("Computing teacher embeddings for comparison...")
        teacher_embs = _embed_texts(
            teacher_model, teacher_tokenizer, texts, device
        )
        logger.info("Computing student embeddings for comparison...")
        student_embs = _embed_texts(
            student_model, student_tokenizer, texts, device
        )
        teacher_embeddings = teacher_embs
        student_embeddings = student_embs

    if teacher_embeddings is not None and student_embeddings is not None:
        result["Similarity"] = {
            "cosine_similarity": round(
                compute_cosine_similarity(teacher_embeddings, student_embeddings), 4
            ),
            "mse": round(
                compute_mse(teacher_embeddings, student_embeddings), 6
            ),
        }

    return result


@torch.no_grad()
def _embed_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Helper: tokenize and embed a list of texts, returning a tensor."""
    model.eval()
    encoded = tokenizer(
        texts,
        max_length=NLPConfig.MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded)
    pooled = mean_pooling(outputs, encoded["attention_mask"])
    return F.normalize(pooled, p=2, dim=1).cpu()
