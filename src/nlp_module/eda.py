"""Exploratory data analysis for the CortexAI NLP module.

Purpose:
    Summarize loaded and preprocessed brain tumor radiology reports for data
    quality checks before embedding generation.
Author:
    Nour Hossam
Dependencies:
    collections, dataclasses, json, pathlib, re, statistics, pandas
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd


EDA_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True, slots=True)
class EDASummary:
    """Compact EDA summary for NLP report data."""

    record_count: int
    empty_text_count: int
    min_char_length: int
    max_char_length: int
    mean_char_length: float
    median_char_length: float
    min_word_count: int
    max_word_count: int
    mean_word_count: float
    median_word_count: float
    top_terms: tuple[tuple[str, int], ...]
    label_distributions: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable EDA summary."""
        return {
            "record_count": self.record_count,
            "empty_text_count": self.empty_text_count,
            "min_char_length": self.min_char_length,
            "max_char_length": self.max_char_length,
            "mean_char_length": self.mean_char_length,
            "median_char_length": self.median_char_length,
            "min_word_count": self.min_word_count,
            "max_word_count": self.max_word_count,
            "mean_word_count": self.mean_word_count,
            "median_word_count": self.median_word_count,
            "top_terms": [
                {"term": term, "count": count} for term, count in self.top_terms
            ],
            "label_distributions": self.label_distributions,
        }


def generate_eda_summary(
    dataframe: pd.DataFrame,
    text_column: str = "clean_text",
    label_columns: tuple[str, ...] = (),
    top_k: int = 25,
) -> EDASummary:
    """Generate an EDA summary for report text.

    Args:
        dataframe: Dataframe containing report text.
        text_column: Text column to summarize.
        label_columns: Optional label columns to count.
        top_k: Number of frequent terms to include.

    Returns:
        EDASummary with length, term, and label statistics.

    Raises:
        ValueError: If text_column is missing.
    """
    if text_column not in dataframe.columns:
        raise ValueError(f"Missing EDA text column: {text_column}")

    texts = dataframe[text_column].fillna("").astype(str)
    char_lengths = texts.map(len).tolist()
    word_counts = texts.map(lambda text: len(text.split())).tolist()
    empty_text_count = int(texts.str.strip().eq("").sum())

    terms: Counter[str] = Counter()
    for text in texts:
        terms.update(_terms(text))

    label_distributions: dict[str, dict[str, int]] = {}
    for label_column in label_columns:
        if label_column in dataframe.columns:
            counts = dataframe[label_column].fillna("<missing>").astype(str).value_counts()
            label_distributions[label_column] = {
                str(label): int(count) for label, count in counts.items()
            }

    return EDASummary(
        record_count=len(dataframe.index),
        empty_text_count=empty_text_count,
        min_char_length=min(char_lengths, default=0),
        max_char_length=max(char_lengths, default=0),
        mean_char_length=float(mean(char_lengths)) if char_lengths else 0.0,
        median_char_length=float(median(char_lengths)) if char_lengths else 0.0,
        min_word_count=min(word_counts, default=0),
        max_word_count=max(word_counts, default=0),
        mean_word_count=float(mean(word_counts)) if word_counts else 0.0,
        median_word_count=float(median(word_counts)) if word_counts else 0.0,
        top_terms=tuple(terms.most_common(top_k)),
        label_distributions=label_distributions,
    )


def save_eda_summary(summary: EDASummary, path: Path) -> Path:
    """Save an EDA summary as JSON.

    Args:
        summary: EDA summary to save.
        path: Destination JSON path.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _terms(text: str) -> list[str]:
    """Extract simple lowercase alphanumeric terms for EDA only."""
    return [
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
        if term not in EDA_STOPWORDS
    ]
