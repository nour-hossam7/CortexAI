
"""Clinical text preprocessing for the CortexAI NLP module.

Purpose:
    Clean brain tumor radiology reports while preserving medical meaning.
Author:
    Nour Hossam
Dependencies:
    re, unicodedata, pandas
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


DEFAULT_MEDICAL_ABBREVIATIONS: dict[str, str] = {
    "r/o": "rule out",
    "w/": "with",
    "w/o": "without",
    "c/o": "complains of",
    "h/o": "history of",
    "hx": "history",
    "dx": "diagnosis",
    "tx": "treatment",
    "rx": "prescription",
    "pt": "patient",
    "pts": "patients",
    "yo": "year old",
    "y/o": "year old",
    "fu": "follow up",
    "f/u": "follow up",
    "s/p": "status post",
    "b/l": "bilateral",
    "lt": "left",
    "rt": "right",
}

PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


@dataclass(slots=True)
class PreprocessingOptions:
    """Options that control clinical report cleaning behavior."""

    lowercase: bool = False
    normalize_abbreviations: bool = True
    missing_text_value: str = ""
    abbreviation_map: Mapping[str, str] = field(
        default_factory=lambda: DEFAULT_MEDICAL_ABBREVIATIONS.copy()
    )


class TextPreprocessor:
    """Normalize and clean clinical report text without stripping terminology."""

    def __init__(self, options: PreprocessingOptions | None = None) -> None:
        """Create a text preprocessor.

        Args:
            options: Optional preprocessing configuration.
        """
        self.options = options or PreprocessingOptions()

    def clean_text(self, value: Any) -> str:
        """Clean one report text value.

        Args:
            value: Raw report text, missing value, or any value convertible to text.

        Returns:
            A normalized clinical text string.
        """
        if _is_missing(value):
            text = self.options.missing_text_value
        else:
            text = str(value)

        text = unicodedata.normalize("NFKC", text)
        text = text.translate(PUNCTUATION_TRANSLATION)
        text = _normalize_whitespace(text)

        if self.options.normalize_abbreviations:
            text = self._expand_abbreviations(text)

        if self.options.lowercase:
            text = text.lower()

        return _normalize_whitespace(text)

    def preprocess_dataframe(
        self,
        dataframe: pd.DataFrame,
        text_column: str = "text",
        output_column: str = "clean_text",
    ) -> pd.DataFrame:
        """Clean report text in a dataframe.

        Args:
            dataframe: Input dataframe containing a report text column.
            text_column: Name of the raw text column.
            output_column: Name of the column that will contain cleaned text.

        Returns:
            A copied dataframe with the cleaned text column added.

        Raises:
            ValueError: If the requested text column is missing.
        """
        if text_column not in dataframe.columns:
            raise ValueError(f"Missing required text column: {text_column}")

        processed = dataframe.copy()
        processed[output_column] = processed[text_column].map(self.clean_text)
        return processed

    def _expand_abbreviations(self, text: str) -> str:
        """Expand configured clinical abbreviations in a text string."""
        normalized = text
        for abbreviation, replacement in self.options.abbreviation_map.items():
            pattern = re.compile(
                rf"(?<!\w){re.escape(abbreviation)}(?!\w)",
                flags=re.IGNORECASE,
            )
            normalized = pattern.sub(replacement, normalized)
        return normalized


def clean_report_text(value: Any, options: PreprocessingOptions | None = None) -> str:
    """Clean one report value using CortexAI clinical preprocessing rules.

    Args:
        value: Raw report text or missing value.
        options: Optional preprocessing configuration.

    Returns:
        Cleaned report text.
    """
    return TextPreprocessor(options=options).clean_text(value)


def preprocess_reports(
    dataframe: pd.DataFrame,
    text_column: str = "text",
    output_column: str = "clean_text",
    options: PreprocessingOptions | None = None,
) -> pd.DataFrame:
    """Clean all reports in a dataframe.

    Args:
        dataframe: Input dataframe with raw report text.
        text_column: Source text column name.
        output_column: Destination cleaned text column name.
        options: Optional preprocessing configuration.

    Returns:
        A dataframe copy containing cleaned reports.
    """
    return TextPreprocessor(options=options).preprocess_dataframe(
        dataframe=dataframe,
        text_column=text_column,
        output_column=output_column,
    )


def _normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _is_missing(value: Any) -> bool:
    """Return True when a value should be treated as missing text."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
