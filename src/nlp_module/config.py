"""Configuration objects for the CortexAI NLP module.

Purpose:
    Keep model choices, preprocessing options, paths, device selection, and
    output names configurable from one production-ready object.
Author:
    Nour Hossam
Dependencies:
    dataclasses, pathlib, typing, torch, src.nlp_module.paths
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .paths import NLPPaths, SUPPORTED_DATASET_SUFFIXES, default_paths
from .preprocessing import DEFAULT_MEDICAL_ABBREVIATIONS


BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
CLINICALBERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

MODEL_ALIASES = {
    "biobert": BIOBERT_MODEL_NAME,
    "clinicalbert": CLINICALBERT_MODEL_NAME,
}

SUPPORTED_POOLING_STRATEGIES = ("cls", "mean")


@dataclass(frozen=True, slots=True)
class NLPConfig:
    """Runtime configuration for the complete NLP pipeline."""

    paths: NLPPaths = field(default_factory=default_paths)
    model_alias: str = "biobert"
    model_name: str | None = None
    tokenizer_name: str | None = None
    pooling_strategy: str = "cls"
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 256
    padding: bool | str = "max_length"
    truncation: bool = True
    random_seed: int = 42
    lowercase: bool = False
    normalize_abbreviations: bool = True
    text_column: str | None = None
    id_column: str | None = None
    label_columns: tuple[str, ...] = ()
    clean_text_column: str = "clean_text"
    supported_suffixes: tuple[str, ...] = SUPPORTED_DATASET_SUFFIXES
    local_files_only: bool = False
    output_clean_reports_filename: str = "cleaned_reports.csv"
    output_tokens_filename: str = "tokenized_inputs.pt"
    output_embeddings_filename: str = "nlp_embeddings.npz"
    output_embeddings_npy_filename: str = "embeddings.npy"
    output_labels_filename: str = "labels.csv"
    output_metadata_filename: str = "metadata.csv"
    output_config_filename: str = "config.json"
    output_manifest_filename: str = "nlp_feature_manifest.json"
    output_eda_filename: str = "nlp_eda_summary.json"
    abbreviation_map: dict[str, str] = field(
        default_factory=lambda: DEFAULT_MEDICAL_ABBREVIATIONS.copy()
    )

    def __post_init__(self) -> None:
        """Validate immutable configuration values after initialization."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.max_length <= 0:
            raise ValueError("max_length must be greater than zero.")
        if self.pooling_strategy not in SUPPORTED_POOLING_STRATEGIES:
            choices = ", ".join(SUPPORTED_POOLING_STRATEGIES)
            raise ValueError(f"pooling_strategy must be one of: {choices}")
        if self.model_alias not in MODEL_ALIASES and self.model_name is None:
            aliases = ", ".join(sorted(MODEL_ALIASES))
            raise ValueError(
                "model_alias must be one of "
                f"{aliases}, or model_name must be provided."
            )
        object.__setattr__(self, "label_columns", tuple(self.label_columns))
        object.__setattr__(
            self,
            "supported_suffixes",
            tuple(suffix.lower() for suffix in self.supported_suffixes),
        )

    def resolved_model_name(self) -> str:
        """Return the Hugging Face model name selected by this config."""
        return self.model_name or MODEL_ALIASES[self.model_alias]

    def resolved_tokenizer_name(self) -> str:
        """Return the tokenizer name selected by this config."""
        return self.tokenizer_name or self.resolved_model_name()

    def resolved_device(self) -> str:
        """Resolve the configured compute device to cpu, cuda, or mps."""
        if self.device != "auto":
            return self.device

        try:
            import torch
        except ImportError:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def with_updates(self, **updates: Any) -> "NLPConfig":
        """Return a copy of the configuration with selected values updated."""
        return replace(self, **updates)


def build_config(
    *,
    project_root: Path | None = None,
    raw_textbrats_dir: Path | None = None,
    processed_nlp_dir: Path | None = None,
    reports_dir: Path | None = None,
    **updates: Any,
) -> NLPConfig:
    """Create an NLPConfig with optional path overrides.

    Args:
        project_root: Optional repository root override.
        raw_textbrats_dir: Optional raw TextBraTS directory override.
        processed_nlp_dir: Optional processed NLP output directory override.
        reports_dir: Optional reports directory override.
        updates: Additional NLPConfig field overrides.

    Returns:
        A validated NLPConfig instance.
    """
    paths = default_paths(project_root=project_root)
    if raw_textbrats_dir or processed_nlp_dir or reports_dir:
        paths = NLPPaths(
            project_root=paths.project_root,
            raw_textbrats_dir=(raw_textbrats_dir or paths.raw_textbrats_dir),
            processed_nlp_dir=(processed_nlp_dir or paths.processed_nlp_dir),
            reports_dir=(reports_dir or paths.reports_dir),
        )
    return NLPConfig(paths=paths, **updates)
