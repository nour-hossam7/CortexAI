"""Saving and loading utilities for CortexAI NLP outputs.

Purpose:
    Persist cleaned reports, tokenized inputs, embeddings, EDA summaries, and
    manifests under datasets/processed/nlp for downstream fusion.
Author:
    Nour Hossam
Dependencies:
    dataclasses, datetime, json, pathlib, numpy, pandas, torch,
    src.nlp_module.config, src.nlp_module.eda, src.nlp_module.embeddings,
    src.nlp_module.tokenizer, src.nlp_module.validator
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import NLPConfig
from .eda import EDASummary, save_eda_summary
from .embeddings import EmbeddingResult
from .logger import get_logger, log_duration
from .tokenizer import TokenizedBatch
from .validator import ValidationReport


@dataclass(frozen=True, slots=True)
class SavedFeaturePaths:
    """Paths written by the NLP saving step."""

    cleaned_reports: Path
    tokenized_inputs: Path
    embeddings: Path
    embeddings_npy: Path
    labels: Path
    metadata: Path
    config_json: Path
    manifest: Path
    eda_summary: Path

    def to_dict(self) -> dict[str, str]:
        """Return JSON-serializable saved output paths."""
        return {
            "cleaned_reports": str(self.cleaned_reports),
            "tokenized_inputs": str(self.tokenized_inputs),
            "embeddings": str(self.embeddings),
            "embeddings_npy": str(self.embeddings_npy),
            "labels": str(self.labels),
            "metadata": str(self.metadata),
            "config_json": str(self.config_json),
            "manifest": str(self.manifest),
            "eda_summary": str(self.eda_summary),
        }


def save_pipeline_outputs(
    *,
    reports: pd.DataFrame,
    tokenized: TokenizedBatch,
    embeddings: EmbeddingResult,
    eda_summary: EDASummary,
    validation: ValidationReport,
    config: NLPConfig,
) -> SavedFeaturePaths:
    """Save all primary NLP pipeline outputs.

    Args:
        reports: Cleaned report dataframe.
        tokenized: Tokenized model inputs.
        embeddings: Extracted report embeddings.
        eda_summary: EDA summary.
        validation: Validation report.
        config: NLP module configuration.

    Returns:
        Paths written by the saving step.
    """
    logger = get_logger()
    output_dir = config.paths.processed_nlp_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with log_duration("NLP feature saving", logger=logger):
        cleaned_reports_path = save_cleaned_reports(
            reports,
            output_dir / config.output_clean_reports_filename,
        )
        tokenized_path = save_tokenized_inputs(
            tokenized,
            output_dir / config.output_tokens_filename,
        )
        embeddings_path = save_embeddings(
            embeddings,
            output_dir / config.output_embeddings_filename,
        )
        embeddings_npy_path = save_embedding_array(
            embeddings,
            output_dir / config.output_embeddings_npy_filename,
        )
        labels_path = save_labels(
            reports,
            output_dir / config.output_labels_filename,
            label_columns=config.label_columns,
        )
        metadata_path = save_metadata(
            reports,
            output_dir / config.output_metadata_filename,
            label_columns=config.label_columns,
        )
        config_path = save_config_json(
            config,
            output_dir / config.output_config_filename,
        )
        eda_path = save_eda_summary(
            eda_summary,
            output_dir / config.output_eda_filename,
        )
        manifest_path = save_manifest(
            output_dir / config.output_manifest_filename,
            config=config,
            validation=validation,
            eda_summary=eda_summary,
            embeddings=embeddings,
            saved_paths={
                "cleaned_reports": cleaned_reports_path,
                "tokenized_inputs": tokenized_path,
                "embeddings": embeddings_path,
                "embeddings_npy": embeddings_npy_path,
                "labels": labels_path,
                "metadata": metadata_path,
                "config_json": config_path,
                "eda_summary": eda_path,
            },
        )

    return SavedFeaturePaths(
        cleaned_reports=cleaned_reports_path,
        tokenized_inputs=tokenized_path,
        embeddings=embeddings_path,
        embeddings_npy=embeddings_npy_path,
        labels=labels_path,
        metadata=metadata_path,
        config_json=config_path,
        manifest=manifest_path,
        eda_summary=eda_path,
    )


def save_cleaned_reports(reports: pd.DataFrame, path: Path) -> Path:
    """Save cleaned reports as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reports.to_csv(path, index=False)
    return path


def save_tokenized_inputs(tokenized: TokenizedBatch, path: Path) -> Path:
    """Save tokenized inputs with torch.save."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to save tokenized inputs.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokenized.to_cpu_dict(), path)
    return path


def save_embeddings(embeddings: EmbeddingResult, path: Path) -> Path:
    """Save report embeddings as a compressed NPZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings.embeddings.astype(np.float32),
        report_ids=np.asarray(embeddings.report_ids, dtype=str),
        texts=np.asarray(embeddings.texts, dtype=str),
        model_name=np.asarray([embeddings.model_name], dtype=str),
        pooling_strategy=np.asarray([embeddings.pooling_strategy], dtype=str),
    )
    return path


def save_embedding_array(embeddings: EmbeddingResult, path: Path) -> Path:
    """Save only the embedding matrix as a NumPy NPY file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.embeddings.astype(np.float32))
    return path


def save_labels(
    reports: pd.DataFrame,
    path: Path,
    label_columns: tuple[str, ...] = (),
) -> Path:
    """Save report labels as CSV for Fusion integration.

    When no label columns are configured, the file still contains report_id so
    row alignment can be verified against embeddings and metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["report_id", *[column for column in label_columns if column in reports]]
    reports.loc[:, columns].to_csv(path, index=False)
    return path


def save_metadata(
    reports: pd.DataFrame,
    path: Path,
    label_columns: tuple[str, ...] = (),
) -> Path:
    """Save report metadata as CSV for Fusion integration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded = set(label_columns)
    preferred_columns = [
        "report_id",
        "source_path",
        "source_format",
        "source_row",
        "relative_path",
        "folder_label",
        "text",
        "clean_text",
    ]
    columns = [
        column
        for column in preferred_columns
        if column in reports.columns and column not in excluded
    ]
    if "report_id" not in columns:
        columns.insert(0, "report_id")
    reports.loc[:, list(dict.fromkeys(columns))].to_csv(path, index=False)
    return path


def save_config_json(config: NLPConfig, path: Path) -> Path:
    """Save pipeline configuration as JSON for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_alias": config.model_alias,
        "model_name": config.resolved_model_name(),
        "tokenizer_name": config.resolved_tokenizer_name(),
        "pooling_strategy": config.pooling_strategy,
        "device": config.resolved_device(),
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "padding": config.padding,
        "truncation": config.truncation,
        "random_seed": config.random_seed,
        "lowercase": config.lowercase,
        "normalize_abbreviations": config.normalize_abbreviations,
        "label_columns": list(config.label_columns),
        "clean_text_column": config.clean_text_column,
        "raw_textbrats_dir": str(config.paths.raw_textbrats_dir),
        "processed_nlp_dir": str(config.paths.processed_nlp_dir),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_embeddings(
    path: Path | None = None,
    *,
    processed_dir: Path | None = None,
) -> EmbeddingResult:
    """Load saved report embeddings from an NPZ file.

    Args:
        path: Optional path to an NPZ or NPY embeddings file.
        processed_dir: Optional processed NLP directory for default discovery.

    Returns:
        EmbeddingResult for fusion or dataset construction.
    """
    if path is None:
        base_dir = processed_dir or NLPConfig().paths.processed_nlp_dir
        npz_path = base_dir / NLPConfig().output_embeddings_filename
        npy_path = base_dir / NLPConfig().output_embeddings_npy_filename
        path = npz_path if npz_path.exists() else npy_path

    if path.suffix.lower() == ".npy":
        return _load_embeddings_from_npy(path)

    with np.load(path, allow_pickle=False) as data:
        embeddings = data["embeddings"].astype(np.float32)
        report_ids = tuple(str(value) for value in data["report_ids"].tolist())
        texts = tuple(str(value) for value in data["texts"].tolist())
        model_name = str(data["model_name"][0])
        pooling_strategy = str(data["pooling_strategy"][0])
    return EmbeddingResult(
        embeddings=embeddings,
        report_ids=report_ids,
        model_name=model_name,
        pooling_strategy=pooling_strategy,
        texts=texts,
    )


def load_labels(
    processed_dir: Path | None = None,
    labels_filename: str = "labels.csv",
) -> pd.DataFrame:
    """Load saved NLP labels for Fusion integration."""
    base_dir = processed_dir or NLPConfig().paths.processed_nlp_dir
    return pd.read_csv(base_dir / labels_filename)


def load_metadata(
    processed_dir: Path | None = None,
    metadata_filename: str = "metadata.csv",
) -> pd.DataFrame:
    """Load saved NLP metadata for Fusion integration."""
    base_dir = processed_dir or NLPConfig().paths.processed_nlp_dir
    return pd.read_csv(base_dir / metadata_filename)


def get_feature_dimension(
    processed_dir: Path | None = None,
    embeddings_filename: str | None = None,
) -> int:
    """Return the saved NLP embedding feature dimension."""
    base_dir = processed_dir or NLPConfig().paths.processed_nlp_dir
    if embeddings_filename is None:
        npy_path = base_dir / NLPConfig().output_embeddings_npy_filename
        npz_path = base_dir / NLPConfig().output_embeddings_filename
        path = npy_path if npy_path.exists() else npz_path
    else:
        path = base_dir / embeddings_filename
    if path.suffix.lower() == ".npz":
        embeddings = load_embeddings(path).embeddings
    else:
        embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embeddings array, got {embeddings.shape}.")
    return int(embeddings.shape[1])


def save_manifest(
    path: Path,
    *,
    config: NLPConfig,
    validation: ValidationReport,
    eda_summary: EDASummary,
    embeddings: EmbeddingResult,
    saved_paths: dict[str, Path],
) -> Path:
    """Save a JSON manifest describing NLP outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_name": config.resolved_model_name(),
        "tokenizer_name": config.resolved_tokenizer_name(),
        "pooling_strategy": config.pooling_strategy,
        "device": config.resolved_device(),
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "padding": config.padding,
        "truncation": config.truncation,
        "clean_text_column": config.clean_text_column,
        "embedding": embeddings.metadata(),
        "validation": validation.to_dict(),
        "eda": eda_summary.to_dict(),
        "outputs": {name: str(output_path) for name, output_path in saved_paths.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_embeddings_from_npy(path: Path) -> EmbeddingResult:
    """Load embeddings from NPY plus metadata/config sidecar files."""
    embeddings = np.load(path).astype(np.float32)
    base_dir = path.parent
    metadata_path = base_dir / NLPConfig().output_metadata_filename
    config_path = base_dir / NLPConfig().output_config_filename

    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if "report_id" in metadata:
            report_ids = tuple(str(value) for value in metadata["report_id"].tolist())
        else:
            report_ids = tuple(str(index) for index in range(embeddings.shape[0]))
        text_column = "clean_text" if "clean_text" in metadata else "text"
        texts = (
            tuple(str(value) for value in metadata[text_column].fillna("").tolist())
            if text_column in metadata
            else ()
        )
    else:
        report_ids = tuple(str(index) for index in range(embeddings.shape[0]))
        texts = ()

    if config_path.exists():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = str(config_payload.get("model_name", "unknown"))
        pooling_strategy = str(config_payload.get("pooling_strategy", "unknown"))
    else:
        model_name = "unknown"
        pooling_strategy = "unknown"

    return EmbeddingResult(
        embeddings=embeddings,
        report_ids=report_ids,
        model_name=model_name,
        pooling_strategy=pooling_strategy,
        texts=texts,
    )
