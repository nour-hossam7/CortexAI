"""High-level NLP feature extraction for CortexAI.

Purpose:
    Coordinate validation, preprocessing, EDA, tokenization, and transformer
    embeddings for in-memory clinical report data.
Author:
    Nour Hossam
Dependencies:
    dataclasses, pandas, src.nlp_module preprocessing, validation,
    tokenization, embedding, and EDA utilities
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import NLPConfig
from .eda import EDASummary, generate_eda_summary
from .embeddings import EmbeddingResult, TransformerEmbedder
from .logger import get_logger, log_duration
from .preprocessing import PreprocessingOptions, TextPreprocessor
from .tokenizer import TokenizedBatch, TransformerTextTokenizer
from .validator import ValidationReport, validate_reports_dataframe


@dataclass(frozen=True, slots=True)
class NLPFeatureBundle:
    """In-memory NLP outputs ready for saving or fusion."""

    reports: pd.DataFrame
    eda_summary: EDASummary
    tokenized: TokenizedBatch
    embeddings: EmbeddingResult
    validation: ValidationReport


class NLPFeatureExtractor:
    """Production feature extractor for radiology report text."""

    def __init__(
        self,
        config: NLPConfig,
        tokenizer: TransformerTextTokenizer | None = None,
        embedder: TransformerEmbedder | None = None,
    ) -> None:
        """Create an NLP feature extractor.

        Args:
            config: NLP module configuration.
            tokenizer: Optional tokenizer wrapper.
            embedder: Optional embedder wrapper.
        """
        self.config = config
        self.logger = get_logger()
        self.tokenizer = tokenizer
        self.embedder = embedder

    def extract(self, reports: pd.DataFrame) -> NLPFeatureBundle:
        """Extract cleaned text, tokens, EDA, and embeddings from reports.

        Args:
            reports: Dataframe with report_id and text columns.

        Returns:
            NLPFeatureBundle containing all in-memory NLP outputs.
        """
        with log_duration("NLP feature extraction", logger=self.logger):
            validation = validate_reports_dataframe(reports)
            validation.raise_for_errors()

            preprocessing_options = PreprocessingOptions(
                lowercase=self.config.lowercase,
                normalize_abbreviations=self.config.normalize_abbreviations,
                abbreviation_map=self.config.abbreviation_map,
            )
            preprocessor = TextPreprocessor(options=preprocessing_options)
            cleaned_reports = preprocessor.preprocess_dataframe(
                reports,
                text_column="text",
                output_column=self.config.clean_text_column,
            )

            eda_summary = generate_eda_summary(
                cleaned_reports,
                text_column=self.config.clean_text_column,
                label_columns=self.config.label_columns,
            )

            active_tokenizer = self.tokenizer or TransformerTextTokenizer(self.config)
            tokenized = active_tokenizer.tokenize_dataframe(
                cleaned_reports,
                text_column=self.config.clean_text_column,
                id_column="report_id",
            )

            active_embedder = self.embedder or TransformerEmbedder(self.config)
            embeddings = active_embedder.embed_tokenized(tokenized)

        return NLPFeatureBundle(
            reports=cleaned_reports,
            eda_summary=eda_summary,
            tokenized=tokenized,
            embeddings=embeddings,
            validation=validation,
        )


def extract_features_from_dataframe(
    reports: pd.DataFrame,
    config: NLPConfig,
) -> NLPFeatureBundle:
    """Extract NLP features from a dataframe using the production pipeline."""
    return NLPFeatureExtractor(config).extract(reports)
