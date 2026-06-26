"""Filesystem path utilities for the CortexAI NLP module.

Purpose:
    Centralize NLP input, output, and report paths without hardcoding them
    across the pipeline.
Author:
    Nour Hossam
Dependencies:
    pathlib, dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_DATASET_SUFFIXES = (".csv", ".json", ".jsonl", ".txt")


@dataclass(frozen=True, slots=True)
class NLPPaths:
    """Resolved filesystem paths used by the NLP module."""

    project_root: Path
    raw_textbrats_dir: Path
    processed_nlp_dir: Path
    reports_dir: Path

    def ensure_output_dirs(self) -> None:
        """Create NLP output and report directories when they are missing."""
        self.processed_nlp_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


def default_project_root() -> Path:
    """Return the CortexAI repository root inferred from this module file."""
    return Path(__file__).resolve().parents[2]


def default_paths(project_root: Path | None = None) -> NLPPaths:
    """Build the default NLP path configuration.

    Args:
        project_root: Optional repository root override.

    Returns:
        NLPPaths with raw TextBraTS, processed NLP, and reports directories.
    """
    root = (project_root or default_project_root()).resolve()
    return NLPPaths(
        project_root=root,
        raw_textbrats_dir=root / "datasets" / "raw" / "textbrats",
        processed_nlp_dir=root / "datasets" / "processed" / "nlp",
        reports_dir=root / "reports",
    )


def is_supported_dataset_file(
    path: Path,
    supported_suffixes: tuple[str, ...] = SUPPORTED_DATASET_SUFFIXES,
) -> bool:
    """Return True when a path is a supported TextBraTS data file."""
    return path.is_file() and path.suffix.lower() in supported_suffixes


def iter_dataset_files(
    root: Path,
    supported_suffixes: tuple[str, ...] = SUPPORTED_DATASET_SUFFIXES,
) -> tuple[Path, ...]:
    """Return supported dataset files under a directory, sorted by path.

    Args:
        root: Directory to scan recursively.
        supported_suffixes: File suffixes accepted by the NLP loader.

    Returns:
        Sorted tuple of supported file paths.
    """
    if not root.exists():
        return ()
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if is_supported_dataset_file(path, supported_suffixes)
        )
    )


def relative_to_root(path: Path, root: Path) -> str:
    """Return a stable relative path string when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
