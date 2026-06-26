"""TextBraTS data inspection and loading for the CortexAI NLP module.

Purpose:
    Inspect raw TextBraTS directories and load CSV, JSON, JSONL, TXT, and
    folder-based report datasets into a stable tabular format.
Author:
    Nour Hossam
Dependencies:
    collections, dataclasses, json, pathlib, pandas,
    src.nlp_module.config, src.nlp_module.logger, src.nlp_module.paths
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import NLPConfig
from .logger import get_logger, log_duration
from .paths import iter_dataset_files, relative_to_root


TEXT_COLUMN_CANDIDATES = (
    "text",
    "report",
    "reports",
    "radiology_report",
    "radiology_text",
    "clinical_text",
    "findings",
    "impression",
    "description",
    "content",
    "sentence",
    "note",
)

ID_COLUMN_CANDIDATES = (
    "report_id",
    "id",
    "case_id",
    "patient_id",
    "subject_id",
    "study_id",
    "accession",
    "filename",
    "file_name",
)


@dataclass(frozen=True, slots=True)
class DatasetInspection:
    """Summary of the raw TextBraTS directory contents."""

    root: Path
    exists: bool
    total_files: int
    supported_files: tuple[Path, ...]
    unsupported_files: tuple[Path, ...]
    files_by_extension: dict[str, int]
    directories: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        """Return True when no supported dataset files are present."""
        return len(self.supported_files) == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable inspection summary."""
        return {
            "root": str(self.root),
            "exists": self.exists,
            "total_files": self.total_files,
            "supported_file_count": len(self.supported_files),
            "unsupported_file_count": len(self.unsupported_files),
            "files_by_extension": dict(self.files_by_extension),
            "directories": list(self.directories),
            "supported_files": [str(path) for path in self.supported_files],
            "unsupported_files": [str(path) for path in self.unsupported_files],
        }


class TextBraTSDataLoader:
    """Load TextBraTS-style clinical report files into a dataframe."""

    def __init__(self, config: NLPConfig) -> None:
        """Create a data loader.

        Args:
            config: NLP module configuration.
        """
        self.config = config
        self.logger = get_logger()

    def inspect(self) -> DatasetInspection:
        """Inspect the configured raw TextBraTS directory."""
        return inspect_textbrats_dataset(self.config)

    def load(self) -> pd.DataFrame:
        """Load all supported raw TextBraTS files into one dataframe.

        Returns:
            Dataframe with canonical report_id, text, source_path,
            source_format, source_row, and relative_path columns.

        Raises:
            FileNotFoundError: If the raw TextBraTS directory does not exist.
            ValueError: If no supported files are found.
        """
        root = self.config.paths.raw_textbrats_dir
        if not root.exists():
            raise FileNotFoundError(f"TextBraTS directory does not exist: {root}")

        files = iter_dataset_files(root, self.config.supported_suffixes)
        if not files:
            raise ValueError(f"No supported TextBraTS files found in: {root}")

        frames: list[pd.DataFrame] = []
        with log_duration("NLP data loading", logger=self.logger):
            for path in files:
                self.logger.info("Loading %s", path)
                frame = self._load_file(path)
                if not frame.empty:
                    frames.append(frame)

        if not frames:
            raise ValueError("Supported TextBraTS files were found but no rows loaded.")

        loaded = pd.concat(frames, ignore_index=True)
        loaded["report_id"] = loaded["report_id"].astype(str)
        return loaded

    def _load_file(self, path: Path) -> pd.DataFrame:
        """Load one supported file and canonicalize its columns."""
        suffix = path.suffix.lower()
        if suffix == ".csv":
            raw_frame = pd.read_csv(path)
            return self._canonicalize_table(raw_frame, path)
        if suffix in {".json", ".jsonl"}:
            raw_frame = _read_json_table(path)
            return self._canonicalize_table(raw_frame, path)
        if suffix == ".txt":
            return self._load_text_file(path)
        raise ValueError(f"Unsupported NLP dataset file type: {path}")

    def _load_text_file(self, path: Path) -> pd.DataFrame:
        """Load one plain-text report file."""
        root = self.config.paths.raw_textbrats_dir
        text = _read_text_file(path)
        relative_path = relative_to_root(path, root)
        row: dict[str, Any] = {
            "report_id": Path(relative_path).with_suffix("").as_posix(),
            "text": text,
            "source_path": str(path),
            "source_format": ".txt",
            "source_row": 0,
            "relative_path": relative_path,
        }

        parent_label = path.parent.name if path.parent != root else None
        if parent_label:
            row["folder_label"] = parent_label

        return pd.DataFrame([row])

    def _canonicalize_table(self, raw_frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        """Convert a CSV or JSON table to canonical NLP columns."""
        frame = raw_frame.copy()
        id_column = _select_id_column(frame, self.config.id_column)
        excluded_text_columns = tuple(
            column
            for column in (id_column, *self.config.label_columns)
            if column is not None
        )
        text_columns = _select_text_columns(
            frame,
            configured_column=self.config.text_column,
            excluded_columns=excluded_text_columns,
        )

        if not text_columns:
            text_values = _combine_textual_columns(frame, excluded_text_columns)
            self.logger.warning(
                "No explicit text column found in %s; combined textual columns.",
                path,
            )
        elif len(text_columns) == 1:
            text_values = frame[text_columns[0]]
        else:
            text_values = _combine_selected_columns(frame, text_columns)

        if id_column is None:
            report_ids = [
                f"{path.stem}_{row_index}" for row_index in range(len(frame.index))
            ]
        else:
            report_ids = frame[id_column].fillna("").astype(str)

        root = self.config.paths.raw_textbrats_dir
        relative_path = relative_to_root(path, root)
        canonical = frame.copy()
        canonical["report_id"] = report_ids
        canonical["text"] = text_values
        canonical["source_path"] = str(path)
        canonical["source_format"] = path.suffix.lower()
        canonical["source_row"] = range(len(canonical.index))
        canonical["relative_path"] = relative_path
        return canonical


def inspect_textbrats_dataset(config: NLPConfig) -> DatasetInspection:
    """Inspect the raw TextBraTS directory described by a config.

    Args:
        config: NLP module configuration.

    Returns:
        DatasetInspection with supported and unsupported files.
    """
    root = config.paths.raw_textbrats_dir
    exists = root.exists()
    all_files = tuple(sorted(path for path in root.rglob("*") if path.is_file())) if exists else ()
    supported = tuple(
        path
        for path in all_files
        if path.suffix.lower() in config.supported_suffixes
        and path.name != ".gitkeep"
    )
    unsupported = tuple(
        path
        for path in all_files
        if path not in supported and path.name != ".gitkeep"
    )
    extension_counts = Counter(path.suffix.lower() or "<none>" for path in supported)
    directories = (
        tuple(sorted(relative_to_root(path, root) for path in root.rglob("*") if path.is_dir()))
        if exists
        else ()
    )
    return DatasetInspection(
        root=root,
        exists=exists,
        total_files=len([path for path in all_files if path.name != ".gitkeep"]),
        supported_files=supported,
        unsupported_files=unsupported,
        files_by_extension=dict(extension_counts),
        directories=directories,
    )


def load_textbrats_dataset(config: NLPConfig) -> pd.DataFrame:
    """Load TextBraTS reports using the configured loader."""
    return TextBraTSDataLoader(config).load()


def _read_text_file(path: Path) -> str:
    """Read a text file with UTF-8 first and a safe fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _read_json_table(path: Path) -> pd.DataFrame:
    """Read JSON or JSONL as a dataframe."""
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        list_key = next(
            (
                key
                for key, value in payload.items()
                if isinstance(value, list) and all(isinstance(item, dict) for item in value)
            ),
            None,
        )
        if list_key is not None:
            return pd.json_normalize(payload[list_key])
        return pd.json_normalize(payload)
    raise ValueError(f"Unsupported JSON structure in {path}")


def _select_text_columns(
    frame: pd.DataFrame,
    configured_column: str | None,
    excluded_columns: tuple[str, ...],
) -> list[str]:
    """Choose report text columns from a dataframe."""
    columns_by_lower = {str(column).lower(): column for column in frame.columns}
    excluded = {column.lower() for column in excluded_columns}
    if configured_column:
        if configured_column not in frame.columns:
            raise ValueError(f"Configured text column not found: {configured_column}")
        return [configured_column]

    selected: list[str] = []
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in columns_by_lower and candidate not in excluded:
            selected.append(str(columns_by_lower[candidate]))

    if selected:
        return list(dict.fromkeys(selected))

    textual_columns = _textual_columns(frame, excluded_columns)
    if len(textual_columns) == 1:
        return textual_columns
    return []


def _select_id_column(frame: pd.DataFrame, configured_column: str | None) -> str | None:
    """Choose the report identifier column from a dataframe."""
    columns_by_lower = {str(column).lower(): column for column in frame.columns}
    if configured_column:
        if configured_column not in frame.columns:
            raise ValueError(f"Configured id column not found: {configured_column}")
        return configured_column

    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in columns_by_lower:
            return str(columns_by_lower[candidate])
    return None


def _combine_textual_columns(
    frame: pd.DataFrame,
    excluded_columns: tuple[str, ...],
) -> pd.Series:
    """Combine textual columns when no explicit text column exists."""
    textual_columns = _textual_columns(frame, excluded_columns)
    if not textual_columns:
        return pd.Series([""] * len(frame.index), index=frame.index)
    return (
        frame[textual_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )


def _combine_selected_columns(
    frame: pd.DataFrame,
    selected_columns: list[str],
) -> pd.Series:
    """Combine selected report text columns in source order."""
    return (
        frame[selected_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )


def _textual_columns(
    frame: pd.DataFrame,
    excluded_columns: tuple[str, ...],
) -> list[str]:
    """Return columns that look textual and are not excluded labels."""
    excluded = {column.lower() for column in excluded_columns}
    textual: list[str] = []
    for column in frame.columns:
        column_name = str(column)
        if column_name.lower() in excluded:
            continue
        series = frame[column]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            textual.append(column_name)
    return textual
