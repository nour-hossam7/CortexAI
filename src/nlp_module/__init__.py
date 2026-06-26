"""Public API for the CortexAI NLP module.

Purpose:
    Expose stable NLP configuration, pipeline, feature extraction, saving, and
    dataset interfaces to the Fusion team without leaking internal details.
Author:
    Nour Hossam
Dependencies:
    src.nlp_module package modules
"""

from __future__ import annotations

from typing import Any

from .config import (
    BIOBERT_MODEL_NAME,
    CLINICALBERT_MODEL_NAME,
    MODEL_ALIASES,
    NLPConfig,
    build_config,
)
from .data_loader import DatasetInspection, TextBraTSDataLoader, load_textbrats_dataset
from .dataset import NLPFeatureDataset
from .embeddings import EmbeddingResult, TransformerEmbedder, extract_embeddings
from .feature_extractor import NLPFeatureBundle, NLPFeatureExtractor
from .preprocessing import PreprocessingOptions, TextPreprocessor, preprocess_reports
from .save_features import (
    SavedFeaturePaths,
    get_feature_dimension,
    load_embeddings,
    load_labels,
    load_metadata,
    save_pipeline_outputs,
)
from .tokenizer import TokenizedBatch, TransformerTextTokenizer, tokenize_reports
from .validator import DatasetValidationError, ValidationReport

__all__ = [
    "BIOBERT_MODEL_NAME",
    "CLINICALBERT_MODEL_NAME",
    "MODEL_ALIASES",
    "DatasetInspection",
    "DatasetValidationError",
    "EmbeddingResult",
    "NLPConfig",
    "NLPFeatureBundle",
    "NLPFeatureDataset",
    "NLPFeatureExtractor",
    "NLPPipeline",
    "NLPPipelineResult",
    "PreprocessingOptions",
    "SavedFeaturePaths",
    "TextBraTSDataLoader",
    "TextPreprocessor",
    "TokenizedBatch",
    "TransformerEmbedder",
    "TransformerTextTokenizer",
    "ValidationReport",
    "build_config",
    "extract_embeddings",
    "extract_features_for_fusion",
    "get_feature_dimension",
    "load_embeddings",
    "load_features_for_fusion",
    "load_labels",
    "load_metadata",
    "load_textbrats_dataset",
    "preprocess_reports",
    "run_textbrats_pipeline",
    "save_pipeline_outputs",
    "tokenize_reports",
]

_LAZY_PIPELINE_EXPORTS = {
    "NLPPipeline",
    "NLPPipelineResult",
    "extract_features_for_fusion",
    "load_features_for_fusion",
    "run_textbrats_pipeline",
}


def __getattr__(name: str) -> Any:
    """Load pipeline exports lazily to keep module execution warning-free."""
    if name in _LAZY_PIPELINE_EXPORTS:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'src.nlp_module' has no attribute {name!r}")
