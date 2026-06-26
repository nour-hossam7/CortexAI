"""End-to-end NLP pipeline and Fusion API for CortexAI.

Purpose:
    Run the complete TextBraTS report pipeline: inspection, validation,
    loading, preprocessing, EDA, tokenization, embedding extraction, saving,
    PyTorch dataset compatibility, and fusion-ready feature access.
Author:
    Nour Hossam
Dependencies:
    argparse, dataclasses, json, pandas, src.nlp_module configuration, loading,
    validation, feature extraction, and saving utilities
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import NLPConfig, build_config
from .data_loader import DatasetInspection, TextBraTSDataLoader
from .embeddings import EmbeddingResult
from .feature_extractor import NLPFeatureBundle, NLPFeatureExtractor
from .logger import configure_logging, get_logger, log_duration
from .save_features import (
    SavedFeaturePaths,
    load_embeddings,
    save_pipeline_outputs,
)
from .validator import (
    ValidationReport,
    merge_validation_reports,
    validate_dataset_inspection,
    validate_reports_dataframe,
    validate_supported_files_readable,
)


@dataclass(frozen=True, slots=True)
class NLPPipelineResult:
    """Complete result returned by a successful NLP pipeline run."""

    inspection: DatasetInspection
    validation: ValidationReport
    features: NLPFeatureBundle
    saved_paths: SavedFeaturePaths


class NLPPipeline:
    """End-to-end TextBraTS NLP processing pipeline."""

    def __init__(
        self,
        config: NLPConfig | None = None,
        feature_extractor: NLPFeatureExtractor | None = None,
    ) -> None:
        """Create the NLP pipeline.

        Args:
            config: Optional NLP module configuration.
            feature_extractor: Optional prebuilt feature extractor.
        """
        self.config = config or NLPConfig()
        self.logger = get_logger()
        self.loader = TextBraTSDataLoader(self.config)
        self.feature_extractor = feature_extractor

    def inspect_dataset(self) -> DatasetInspection:
        """Inspect the configured raw TextBraTS dataset."""
        inspection = self.loader.inspect()
        self.logger.info(
            "TextBraTS inspection: %s supported file(s), %s unsupported file(s).",
            len(inspection.supported_files),
            len(inspection.unsupported_files),
        )
        return inspection

    def run(self) -> NLPPipelineResult:
        """Run the complete NLP pipeline and save processed outputs.

        Returns:
            NLPPipelineResult containing validation, in-memory features, and paths.
        """
        self.config.paths.ensure_output_dirs()
        with log_duration("Complete NLP pipeline", logger=self.logger):
            inspection = self.inspect_dataset()
            inspection_validation = validate_dataset_inspection(inspection)
            for warning in inspection_validation.warnings:
                self.logger.warning(warning)
            inspection_validation.raise_for_errors()

            readable_validation = validate_supported_files_readable(inspection)
            readable_validation.raise_for_errors()

            loaded_reports = self.loader.load()
            data_validation = validate_reports_dataframe(
                loaded_reports,
                label_columns=self.config.label_columns,
                metadata_columns=(
                    "source_path",
                    "source_format",
                    "source_row",
                    "relative_path",
                ),
            )
            for warning in data_validation.warnings:
                self.logger.warning(warning)
            data_validation.raise_for_errors()

            extractor = self.feature_extractor or NLPFeatureExtractor(self.config)
            features = extractor.extract(loaded_reports)
            validation = merge_validation_reports(
                inspection_validation,
                readable_validation,
                data_validation,
                features.validation,
            )

            saved_paths = save_pipeline_outputs(
                reports=features.reports,
                tokenized=features.tokenized,
                embeddings=features.embeddings,
                eda_summary=features.eda_summary,
                validation=validation,
                config=self.config,
            )

        return NLPPipelineResult(
            inspection=inspection,
            validation=validation,
            features=features,
            saved_paths=saved_paths,
        )


def run_textbrats_pipeline(config: NLPConfig | None = None) -> NLPPipelineResult:
    """Run the configured TextBraTS NLP pipeline."""
    return NLPPipeline(config=config).run()


def extract_features_for_fusion(
    reports: pd.DataFrame | Iterable[str],
    config: NLPConfig | None = None,
    report_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return fusion-ready NLP embedding features.

    Args:
        reports: Dataframe with report_id/text columns or an iterable of texts.
        config: Optional NLP module configuration.
        report_ids: Optional IDs when reports is an iterable of strings.

    Returns:
        Dataframe with report_id and nlp_feature_* columns.
    """
    active_config = config or NLPConfig()
    report_frame = _coerce_reports_for_fusion(reports, report_ids=report_ids)
    bundle = NLPFeatureExtractor(active_config).extract(report_frame)
    return bundle.embeddings.to_feature_frame()


def load_features_for_fusion(
    processed_dir: Path | None = None,
    embeddings_filename: str | None = None,
) -> pd.DataFrame:
    """Load saved NLP embeddings as a fusion-ready dataframe.

    Args:
        processed_dir: Optional processed NLP directory.
        embeddings_filename: Embedding NPZ file name.

    Returns:
        Dataframe with report_id and nlp_feature_* columns.
    """
    base_dir = processed_dir or NLPConfig().paths.processed_nlp_dir
    embedding_path = base_dir / embeddings_filename if embeddings_filename else None
    embedding_result: EmbeddingResult = load_embeddings(
        embedding_path,
        processed_dir=base_dir,
    )
    return embedding_result.to_feature_frame()


def _coerce_reports_for_fusion(
    reports: pd.DataFrame | Iterable[str],
    report_ids: Iterable[str] | None,
) -> pd.DataFrame:
    """Normalize Fusion API inputs into the canonical report dataframe."""
    if isinstance(reports, pd.DataFrame):
        return reports.copy()

    texts = [str(text) for text in reports]
    ids = (
        [str(report_id) for report_id in report_ids]
        if report_ids is not None
        else [str(index) for index in range(len(texts))]
    )
    if len(ids) != len(texts):
        raise ValueError("report_ids length must match reports length.")
    return pd.DataFrame({"report_id": ids, "text": texts})


def main() -> None:
    """Command-line entry point for inspecting or running the NLP pipeline."""
    parser = argparse.ArgumentParser(description="Run CortexAI NLP pipeline.")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--model-alias", choices=("biobert", "clinicalbert"), default="biobert")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    configure_logging()
    config = build_config(
        raw_textbrats_dir=args.raw_dir,
        processed_nlp_dir=args.processed_dir,
        model_alias=args.model_alias,
        model_name=args.model_name,
        pooling_strategy=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )

    pipeline = NLPPipeline(config)
    if args.inspect_only:
        inspection = pipeline.inspect_dataset()
        print(json.dumps(inspection.to_dict(), indent=2))
        return

    result = pipeline.run()
    print(json.dumps(result.saved_paths.to_dict(), indent=2))


if __name__ == "__main__":
    main()
