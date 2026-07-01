"""Project paths and artifact discovery with graceful fallbacks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved paths to models, datasets, and reports."""

    root: Path

    @property
    def segresnet_checkpoint(self) -> Path:
        return self.root / "models" / "segmentation" / "best_model.pth"

    @property
    def fusion_checkpoint(self) -> Path:
        return self.root / "models" / "fusion" / "best_decision_model.pth"

    @property
    def clinical_scaler(self) -> Path:
        return self.root / "models" / "fusion" / "clinical_scaler.pkl"

    @property
    def severity_thresholds(self) -> Path:
        return self.root / "models" / "fusion" / "severity_thresholds.json"

    @property
    def dataset_split(self) -> Path:
        return self.root / "datasets" / "splits" / "dataset_split.json"

    @property
    def dataset_info(self) -> Path:
        return self.root / "datasets" / "splits" / "dataset_info.json"

    @property
    def cv_results(self) -> Path:
        return self.root / "reports" / "results"

    @property
    def cv_figures(self) -> Path:
        return self.root / "reports" / "figures"

    @property
    def fusion_figures(self) -> Path:
        return self.root / "reports" / "figures" / "fusion"

    @property
    def fusion_results(self) -> Path:
        return self.root / "reports" / "results" / "fusion"

    @property
    def fusion_repr(self) -> Path:
        return self.root / "reports" / "fusion" / "representations"

    @property
    def nlp_embeddings_dir(self) -> Path:
        return self.root / "datasets" / "processed" / "nlp" / "biobert-base-cased-v1.1"

    @property
    def clinicalbert_dir(self) -> Path:
        return self.root / "datasets" / "processed" / "nlp" / "Bio_ClinicalBERT"

    @property
    def clinical_features_dir(self) -> Path:
        return self.root / "datasets" / "processed" / "clinical_features"

    def exists(self, path: Path) -> bool:
        return path.exists()

    def list_figures(self, directory: Path, pattern: str = "*.png") -> list[Path]:
        if not directory.exists():
            return []
        return sorted(directory.glob(pattern))


@lru_cache(maxsize=1)
def get_paths() -> ArtifactPaths:
    return ArtifactPaths(root=PROJECT_ROOT)


@lru_cache(maxsize=1)
def load_dataset_info() -> dict:
    path = get_paths().dataset_info
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"patients": {"total": 369, "train": 257, "validation": 56, "test": 56}}
