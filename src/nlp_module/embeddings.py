"""Transformer embedding extraction for the CortexAI NLP module.

Purpose:
    Generate BioBERT or ClinicalBERT report embeddings with CLS or mean
    pooling for downstream fusion.
Author:
    Nour Hossam
Dependencies:
    dataclasses, numpy, torch, transformers, src.nlp_module.config,
    src.nlp_module.tokenizer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .config import NLPConfig
from .logger import get_logger, log_duration
from .tokenizer import TokenizedBatch, TransformerTextTokenizer


class TransformerModelProtocol(Protocol):
    """Protocol for Hugging Face-compatible transformer models."""

    def eval(self) -> Any:
        """Switch the model to evaluation mode."""

    def to(self, device: str) -> Any:
        """Move the model to a device."""

    def __call__(self, **kwargs: Any) -> Any:
        """Run a forward pass."""


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Report embeddings and metadata for fusion integration."""

    embeddings: np.ndarray
    report_ids: tuple[str, ...]
    model_name: str
    pooling_strategy: str
    texts: tuple[str, ...] = ()

    def to_feature_frame(self, prefix: str = "nlp_feature_") -> Any:
        """Return embeddings as a pandas dataframe with report IDs."""
        import pandas as pd

        feature_columns = [
            f"{prefix}{index}" for index in range(self.embeddings.shape[1])
        ]
        frame = pd.DataFrame(self.embeddings, columns=feature_columns)
        frame.insert(0, "report_id", list(self.report_ids))
        return frame

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable embedding metadata."""
        return {
            "model_name": self.model_name,
            "pooling_strategy": self.pooling_strategy,
            "embedding_shape": list(self.embeddings.shape),
            "report_count": len(self.report_ids),
        }


class TransformerEmbedder:
    """Generate transformer embeddings from tokenized radiology reports."""

    def __init__(
        self,
        config: NLPConfig,
        model: TransformerModelProtocol | None = None,
    ) -> None:
        """Create a transformer embedder.

        Args:
            config: NLP module configuration.
            model: Optional preloaded Hugging Face-compatible model.
        """
        self.config = config
        self.logger = get_logger()
        self.device = config.resolved_device()
        self.model = model or self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def embed_tokenized(self, tokenized: TokenizedBatch) -> EmbeddingResult:
        """Generate embeddings from tokenized model inputs.

        Args:
            tokenized: Tokenized report inputs.

        Returns:
            EmbeddingResult aligned to report IDs.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for embedding extraction.") from exc

        batches: list[np.ndarray] = []
        total = tokenized.input_ids.shape[0]

        with log_duration("NLP embedding generation", logger=self.logger):
            with torch.no_grad():
                for start in range(0, total, self.config.batch_size):
                    end = start + self.config.batch_size
                    model_inputs = {
                        key: value[start:end].to(self.device)
                        for key, value in tokenized.as_model_inputs().items()
                    }
                    outputs = self.model(**model_inputs)
                    hidden_state = _last_hidden_state(outputs)
                    pooled = _pool_hidden_state(
                        hidden_state=hidden_state,
                        attention_mask=model_inputs["attention_mask"],
                        strategy=self.config.pooling_strategy,
                    )
                    batches.append(pooled.detach().cpu().numpy())

        embeddings = (
            np.concatenate(batches, axis=0).astype(np.float32)
            if batches
            else np.empty((0, 0), dtype=np.float32)
        )
        return EmbeddingResult(
            embeddings=embeddings,
            report_ids=tokenized.report_ids,
            model_name=self.config.resolved_model_name(),
            pooling_strategy=self.config.pooling_strategy,
            texts=tokenized.texts,
        )

    def embed_texts(
        self,
        texts: list[str] | tuple[str, ...],
        report_ids: list[str] | tuple[str, ...] | None = None,
        tokenizer: TransformerTextTokenizer | None = None,
    ) -> EmbeddingResult:
        """Tokenize and embed raw or cleaned report texts.

        Args:
            texts: Report text values.
            report_ids: Optional aligned report IDs.
            tokenizer: Optional preconfigured tokenizer wrapper.

        Returns:
            EmbeddingResult for the reports.
        """
        active_tokenizer = tokenizer or TransformerTextTokenizer(self.config)
        tokenized = active_tokenizer.tokenize_texts(texts=texts, report_ids=report_ids)
        return self.embed_tokenized(tokenized)

    def _load_model(self) -> TransformerModelProtocol:
        """Load the configured transformer model from Hugging Face."""
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for embedding extraction. "
                "Install project requirements before running the NLP pipeline."
            ) from exc

        model_name = self.config.resolved_model_name()
        self.logger.info("Loading embedding model: %s on %s", model_name, self.device)
        try:
            return AutoModel.from_pretrained(
                model_name,
                local_files_only=self.config.local_files_only,
            )
        except OSError as exc:
            raise OSError(
                f"Failed to load embedding model '{model_name}'. "
                "Verify network access to Hugging Face, install CA certificates, "
                "or set model_name to a local model directory and use "
                "local_files_only=True when models are cached offline."
            ) from exc


def extract_embeddings(
    tokenized: TokenizedBatch,
    config: NLPConfig,
) -> EmbeddingResult:
    """Generate embeddings for an existing tokenized batch."""
    return TransformerEmbedder(config).embed_tokenized(tokenized)


def _last_hidden_state(outputs: Any) -> Any:
    """Extract last hidden state from Hugging Face model outputs."""
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs[0]


def _pool_hidden_state(hidden_state: Any, attention_mask: Any, strategy: str) -> Any:
    """Pool token-level hidden states into one vector per report."""
    if strategy == "cls":
        return hidden_state[:, 0, :]
    if strategy == "mean":
        mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
        summed = (hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    raise ValueError(f"Unsupported pooling strategy: {strategy}")
