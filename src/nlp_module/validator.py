"""Validation utilities for the CortexAI NLP module.

Purpose:
    Validate raw TextBraTS inspection results and loaded report data before
    preprocessing, tokenization, and embedding extraction.
Author:
    Nour Hossam
Dependencies:
    dataclasses, pandas, src.nlp_module.data_loader
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import DatasetInspection


class DatasetValidationError(ValueError):
    """Raised when an NLP dataset cannot be processed safely."""


@dataclass(slots=True)
class ValidationReport:
    """Structured validation outcome for data inspection and dataframes."""

    record_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    empty_text_count: int = 0
    duplicate_id_count: int = 0
    missing_id_count: int = 0
    missing_label_count: int = 0
    corrupted_file_count: int = 0
    supported_file_count: int = 0
    unsupported_file_count: int = 0
    missing_label_columns: list[str] = field(default_factory=list)
    missing_metadata_columns: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True when validation has no blocking errors."""
        return not self.errors

    def add_error(self, message: str) -> None:
        """Add a blocking validation error."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a non-blocking validation warning."""
        self.warnings.append(message)

    def raise_for_errors(self) -> None:
        """Raise DatasetValidationError when blocking errors exist."""
        if self.errors:
            raise DatasetValidationError("; ".join(self.errors))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable validation report."""
        return {
            "is_valid": self.is_valid,
            "record_count": self.record_count,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "empty_text_count": self.empty_text_count,
            "duplicate_id_count": self.duplicate_id_count,
            "missing_id_count": self.missing_id_count,
            "missing_label_count": self.missing_label_count,
            "corrupted_file_count": self.corrupted_file_count,
            "supported_file_count": self.supported_file_count,
            "unsupported_file_count": self.unsupported_file_count,
            "missing_label_columns": list(self.missing_label_columns),
            "missing_metadata_columns": list(self.missing_metadata_columns),
        }


def validate_dataset_inspection(inspection: DatasetInspection) -> ValidationReport:
    """Validate raw TextBraTS directory inspection results.

    Args:
        inspection: DatasetInspection returned by the loader.

    Returns:
        ValidationReport for the raw dataset directory.
    """
    report = ValidationReport(
        supported_file_count=len(inspection.supported_files),
        unsupported_file_count=len(inspection.unsupported_files),
    )

    if not inspection.exists:
        report.add_error(f"TextBraTS directory does not exist: {inspection.root}")
        return report

    if inspection.is_empty:
        report.add_error(
            "No TextBraTS data files found. Supported formats: CSV, JSON, JSONL, TXT."
        )

    if inspection.unsupported_files:
        report.add_warning(
            f"Ignoring {len(inspection.unsupported_files)} unsupported file(s)."
        )

    return report


def validate_supported_files_readable(
    inspection: DatasetInspection,
) -> ValidationReport:
    """Validate that supported raw dataset files can be opened and parsed.

    Args:
        inspection: DatasetInspection containing supported file paths.

    Returns:
        ValidationReport with corrupted or unreadable files reported as errors.
    """
    report = ValidationReport(supported_file_count=len(inspection.supported_files))
    for path in inspection.supported_files:
        try:
            _validate_file_readable(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            report.corrupted_file_count += 1
            report.add_error(f"Unreadable or corrupted file: {path} ({exc})")
    return report


def validate_reports_dataframe(
    dataframe: pd.DataFrame,
    text_column: str = "text",
    id_column: str = "report_id",
    label_columns: tuple[str, ...] = (),
    metadata_columns: tuple[str, ...] = (),
) -> ValidationReport:
    """Validate loaded clinical report records.

    Args:
        dataframe: Loaded reports dataframe.
        text_column: Name of the report text column.
        id_column: Name of the unique report identifier column.
        label_columns: Optional label columns that must be present and filled.
        metadata_columns: Optional metadata columns that must be present.

    Returns:
        ValidationReport for the dataframe.
    """
    report = ValidationReport(record_count=len(dataframe.index))

    if dataframe.empty:
        report.add_error("Loaded TextBraTS dataframe is empty.")
        return report

    for required_column in (id_column, text_column):
        if required_column not in dataframe.columns:
            report.add_error(f"Missing required column: {required_column}")

    if report.errors:
        return report

    missing_metadata_columns = [
        column for column in metadata_columns if column not in dataframe.columns
    ]
    if missing_metadata_columns:
        report.missing_metadata_columns.extend(missing_metadata_columns)
        report.add_error(
            "Missing metadata column(s): " + ", ".join(missing_metadata_columns)
        )

    missing_label_columns = [
        column for column in label_columns if column not in dataframe.columns
    ]
    if missing_label_columns:
        report.missing_label_columns.extend(missing_label_columns)
        report.add_error("Missing label column(s): " + ", ".join(missing_label_columns))

    if report.errors:
        return report

    text_values = dataframe[text_column]
    id_values = dataframe[id_column]

    empty_text_mask = text_values.isna() | text_values.astype(str).str.strip().eq("")
    report.empty_text_count = int(empty_text_mask.sum())
    if report.empty_text_count == len(dataframe.index):
        report.add_error("All loaded reports have empty text.")
    elif report.empty_text_count:
        report.add_warning(f"{report.empty_text_count} report(s) have empty text.")

    missing_id_mask = id_values.isna() | id_values.astype(str).str.strip().eq("")
    report.missing_id_count = int(missing_id_mask.sum())
    if report.missing_id_count:
        report.add_error(f"{report.missing_id_count} report(s) have missing IDs.")

    duplicate_mask = id_values.astype(str).duplicated(keep=False)
    report.duplicate_id_count = int(duplicate_mask.sum())
    if report.duplicate_id_count:
        report.add_error(
            f"{report.duplicate_id_count} report(s) share duplicate IDs."
        )

    for label_column in label_columns:
        missing_label_mask = (
            dataframe[label_column].isna()
            | dataframe[label_column].astype(str).str.strip().eq("")
        )
        missing_count = int(missing_label_mask.sum())
        report.missing_label_count += missing_count
        if missing_count:
            report.add_error(
                f"{missing_count} report(s) have missing labels in {label_column}."
            )

    return report


def merge_validation_reports(*reports: ValidationReport) -> ValidationReport:
    """Merge multiple validation reports into a single report."""
    merged = ValidationReport()
    for report in reports:
        merged.record_count = max(merged.record_count, report.record_count)
        merged.empty_text_count += report.empty_text_count
        merged.duplicate_id_count += report.duplicate_id_count
        merged.missing_id_count += report.missing_id_count
        merged.missing_label_count += report.missing_label_count
        merged.corrupted_file_count += report.corrupted_file_count
        merged.supported_file_count += report.supported_file_count
        merged.unsupported_file_count += report.unsupported_file_count
        merged.missing_label_columns.extend(report.missing_label_columns)
        merged.missing_metadata_columns.extend(report.missing_metadata_columns)
        merged.errors.extend(report.errors)
        merged.warnings.extend(report.warnings)
    return merged


def _validate_file_readable(path: Path) -> None:
    """Open and minimally parse a supported dataset file."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        pd.read_csv(path, nrows=5)
        return
    if suffix == ".jsonl":
        pd.read_json(path, lines=True, nrows=5)
        return
    if suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
        return
    if suffix == ".txt":
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            path.read_text(encoding="latin-1")
        return
    raise ValueError(f"Unsupported file type: {path.suffix}")
