"""PyTorch dataset interfaces for CortexAI NLP features.

Purpose:
    Expose tokenized reports and saved text embeddings to training,
    evaluation, and fusion modules through a stable Dataset API.
Author:
    Nour Hossam
Dependencies:
    pathlib, numpy, torch, src.nlp_module.save_features,
    src.nlp_module.tokenizer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .embeddings import EmbeddingResult
from .save_features import load_embeddings
from .tokenizer import TokenizedBatch


class NLPFeatureDataset(Dataset):
    """PyTorch dataset for tokenized reports and optional embeddings."""

    def __init__(
        self,
        report_ids: list[str] | tuple[str, ...],
        tokenized: TokenizedBatch | None = None,
        embeddings: np.ndarray | torch.Tensor | None = None,
        labels: list[Any] | tuple[Any, ...] | np.ndarray | torch.Tensor | None = None,
        texts: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Create an NLP feature dataset.

        Args:
            report_ids: Report identifiers.
            tokenized: Optional tokenized transformer inputs.
            embeddings: Optional report embeddings.
            labels: Optional labels aligned to report IDs.
            texts: Optional report text aligned to report IDs.
        """
        self.report_ids = tuple(str(report_id) for report_id in report_ids)
        self.tokenized = tokenized
        self.embeddings = _to_tensor(embeddings) if embeddings is not None else None
        self.labels = _normalize_labels(labels) if labels is not None else None
        self.texts = tuple(texts) if texts is not None else None
        self._validate_lengths()

    def __len__(self) -> int:
        """Return the number of NLP samples."""
        return len(self.report_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample dictionary for model or fusion consumption."""
        sample: dict[str, Any] = {"report_id": self.report_ids[index]}

        if self.texts is not None:
            sample["text"] = self.texts[index]

        if self.tokenized is not None:
            sample["input_ids"] = self.tokenized.input_ids[index]
            sample["attention_mask"] = self.tokenized.attention_mask[index]
            if self.tokenized.token_type_ids is not None:
                sample["token_type_ids"] = self.tokenized.token_type_ids[index]

        if self.embeddings is not None:
            sample["embedding"] = self.embeddings[index]

        if self.labels is not None:
            sample["label"] = self.labels[index]

        return sample

    @classmethod
    def from_embedding_result(
        cls,
        embedding_result: EmbeddingResult,
        labels: list[Any] | tuple[Any, ...] | np.ndarray | torch.Tensor | None = None,
    ) -> "NLPFeatureDataset":
        """Build a dataset directly from an EmbeddingResult."""
        return cls(
            report_ids=embedding_result.report_ids,
            embeddings=embedding_result.embeddings,
            labels=labels,
            texts=embedding_result.texts or None,
        )

    @classmethod
    def from_processed_dir(
        cls,
        processed_dir: Path,
        embeddings_filename: str = "nlp_embeddings.npz",
        labels: list[Any] | tuple[Any, ...] | np.ndarray | torch.Tensor | None = None,
    ) -> "NLPFeatureDataset":
        """Load saved embeddings from a processed NLP directory."""
        embedding_result = load_embeddings(processed_dir / embeddings_filename)
        return cls.from_embedding_result(embedding_result, labels=labels)

    def _validate_lengths(self) -> None:
        """Validate that all provided arrays are aligned to report IDs."""
        expected = len(self.report_ids)
        if self.tokenized is not None and len(self.tokenized.report_ids) != expected:
            raise ValueError("tokenized length does not match report_ids length.")
        if self.embeddings is not None and self.embeddings.shape[0] != expected:
            raise ValueError("embeddings length does not match report_ids length.")
        if self.labels is not None and len(self.labels) != expected:
            raise ValueError("labels length does not match report_ids length.")
        if self.texts is not None and len(self.texts) != expected:
            raise ValueError("texts length does not match report_ids length.")


def _to_tensor(values: Any) -> torch.Tensor:
    """Convert supported numeric arrays to torch tensors."""
    if isinstance(values, torch.Tensor):
        return values
    array = np.asarray(values)
    return torch.as_tensor(array)


def _normalize_labels(values: Any) -> torch.Tensor | tuple[Any, ...]:
    """Keep string labels intact and convert numeric labels to tensors."""
    if isinstance(values, torch.Tensor):
        return values
    array = np.asarray(values)
    if array.dtype.kind in {"U", "S", "O"}:
        return tuple(array.tolist())
    return torch.as_tensor(array)
