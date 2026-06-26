"""Transformer tokenization for the CortexAI NLP module.

Purpose:
    Tokenize radiology reports with configurable BioBERT or ClinicalBERT
    tokenizers before embedding extraction.
Author:
    Nour Hossam
Dependencies:
    dataclasses, pandas, torch, transformers, src.nlp_module.config
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from .config import NLPConfig
from .logger import get_logger, log_duration


class TokenizerProtocol(Protocol):
    """Protocol for Hugging Face-compatible tokenizers."""

    pad_token_id: int | None

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        """Tokenize a batch of texts."""


@dataclass(frozen=True, slots=True)
class TokenizedBatch:
    """Tokenized model inputs aligned to report IDs."""

    input_ids: Any
    attention_mask: Any
    token_type_ids: Any | None
    report_ids: tuple[str, ...]
    texts: tuple[str, ...]

    def as_model_inputs(self) -> dict[str, Any]:
        """Return tensors accepted by transformer models."""
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        if self.token_type_ids is not None:
            inputs["token_type_ids"] = self.token_type_ids
        return inputs

    def to_cpu_dict(self) -> dict[str, Any]:
        """Return a CPU tensor dictionary suitable for torch.save."""
        inputs = {
            "input_ids": self.input_ids.detach().cpu(),
            "attention_mask": self.attention_mask.detach().cpu(),
            "report_ids": list(self.report_ids),
            "texts": list(self.texts),
        }
        if self.token_type_ids is not None:
            inputs["token_type_ids"] = self.token_type_ids.detach().cpu()
        return inputs


class TransformerTextTokenizer:
    """Configurable BioBERT or ClinicalBERT tokenizer wrapper."""

    def __init__(
        self,
        config: NLPConfig,
        tokenizer: TokenizerProtocol | None = None,
    ) -> None:
        """Create a transformer tokenizer.

        Args:
            config: NLP module configuration.
            tokenizer: Optional preloaded Hugging Face-compatible tokenizer.
        """
        self.config = config
        self.logger = get_logger()
        self.tokenizer = tokenizer or self._load_tokenizer()

    def tokenize_texts(
        self,
        texts: list[str] | tuple[str, ...],
        report_ids: list[str] | tuple[str, ...] | None = None,
    ) -> TokenizedBatch:
        """Tokenize a sequence of report texts.

        Args:
            texts: Cleaned report texts.
            report_ids: Optional report IDs aligned with the texts.

        Returns:
            TokenizedBatch containing tensors and metadata.
        """
        normalized_texts = ["" if text is None else str(text) for text in texts]
        normalized_ids = (
            tuple(str(report_id) for report_id in report_ids)
            if report_ids is not None
            else tuple(str(index) for index in range(len(normalized_texts)))
        )
        if len(normalized_ids) != len(normalized_texts):
            raise ValueError("report_ids length must match texts length.")

        with log_duration("NLP tokenization", logger=self.logger):
            encoded_batches = []
            for start in range(0, len(normalized_texts), self.config.batch_size):
                batch_texts = normalized_texts[start : start + self.config.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.config.max_length,
                    padding=self.config.padding,
                    truncation=self.config.truncation,
                    return_tensors="pt",
                )
                encoded_batches.append(encoded)

        merged = _merge_encoded_batches(
            encoded_batches,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return TokenizedBatch(
            input_ids=merged["input_ids"],
            attention_mask=merged["attention_mask"],
            token_type_ids=merged.get("token_type_ids"),
            report_ids=normalized_ids,
            texts=tuple(normalized_texts),
        )

    def tokenize_dataframe(
        self,
        dataframe: pd.DataFrame,
        text_column: str = "clean_text",
        id_column: str = "report_id",
    ) -> TokenizedBatch:
        """Tokenize report text from a dataframe.

        Args:
            dataframe: Dataframe containing cleaned report text.
            text_column: Name of the cleaned text column.
            id_column: Name of the report ID column.

        Returns:
            TokenizedBatch aligned to dataframe rows.
        """
        if text_column not in dataframe.columns:
            raise ValueError(f"Missing tokenization text column: {text_column}")
        if id_column not in dataframe.columns:
            raise ValueError(f"Missing tokenization id column: {id_column}")
        return self.tokenize_texts(
            texts=dataframe[text_column].fillna("").astype(str).tolist(),
            report_ids=dataframe[id_column].astype(str).tolist(),
        )

    def _load_tokenizer(self) -> TokenizerProtocol:
        """Load the configured tokenizer from Hugging Face Transformers."""
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for NLP tokenization. "
                "Install project requirements before running the NLP pipeline."
            ) from exc

        tokenizer_name = self.config.resolved_tokenizer_name()
        self.logger.info("Loading tokenizer: %s", tokenizer_name)
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=self.config.local_files_only,
        )


def tokenize_reports(
    dataframe: pd.DataFrame,
    config: NLPConfig,
    text_column: str | None = None,
) -> TokenizedBatch:
    """Tokenize cleaned reports with the configured transformer tokenizer."""
    return TransformerTextTokenizer(config).tokenize_dataframe(
        dataframe=dataframe,
        text_column=text_column or config.clean_text_column,
    )


def _merge_encoded_batches(
    encoded_batches: list[dict[str, Any]],
    pad_token_id: int | None,
) -> dict[str, Any]:
    """Merge tokenizer output batches into padded tensors."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for tokenization tensors.") from exc

    if not encoded_batches:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "attention_mask": torch.empty((0, 0), dtype=torch.long),
        }

    merged: dict[str, Any] = {}
    all_keys = set().union(*(batch.keys() for batch in encoded_batches))
    for key in all_keys:
        rows = []
        for batch in encoded_batches:
            if key not in batch:
                continue
            tensor = batch[key]
            rows.extend(tensor.detach().cpu())
        if not rows:
            continue
        padding_value = pad_token_id or 0 if key == "input_ids" else 0
        merged[key] = torch.nn.utils.rnn.pad_sequence(
            rows,
            batch_first=True,
            padding_value=padding_value,
        )
    return merged
