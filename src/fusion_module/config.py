"""
Configuration for the CortexAI Fusion Module.

All paths and hyperparameters live here.
Change TEXT_ENCODER to switch between biobert / clinicalbert.

Author: Ammar Kamal
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FusionConfig:

    # ── Encoder choice ────────────────────────────────────────────────────────
    # Must match the encoder used in NB01 / NB03 / NB04.
    # "biobert"      → datasets/processed/fusion/biobert/
    # "clinicalbert" → datasets/processed/fusion/clinicalbert/
    TEXT_ENCODER: str = "biobert"

    # ── Feature dimensions ────────────────────────────────────────────────────
    IMAGE_DIM: int = 256    # SegResNet bottleneck → global avg pool
    TEXT_DIM:  int = 768    # BioBERT / ClinicalBERT [CLS] token

    # ── Paths — CV raw features (produced by extract_all_features.py) ─────────
    CV_RESULTS_DIR: Path = Path("reports/results")

    # ── Paths — NLP embeddings (produced by nlp_module pipeline) ─────────────
    BIOBERT_NLP_DIR:       Path = Path("datasets/processed/nlp/biobert-base-cased-v1.1")
    CLINICALBERT_NLP_DIR:  Path = Path("datasets/processed/nlp/Bio_ClinicalBERT")

    # ── Paths — fused datasets (produced by data_preparation.py) ─────────────
    FUSION_BASE_DIR: Path = Path("datasets/processed/fusion")

    # ── Paths — clinical features (produced by clinical_features.py) ──────────
    CLINICAL_DIR: Path = Path("datasets/processed/clinical_features")

    # ── Paths — reports ───────────────────────────────────────────────────────
    REPORT_FILE: Path = Path("reports/nlp/textbrats_dataset_inventory.csv")

    # ── Paths — model artifacts ───────────────────────────────────────────────
    MODEL_DIR: Path = Path("models/fusion")
    REPR_DIR:  Path = Path("reports/fusion/representations")

    # ── Training hyperparameters ──────────────────────────────────────────────
    BATCH_SIZE:    int   = 32
    NUM_EPOCHS:    int   = 50
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY:  float = 1e-5
    PATIENCE:      int   = 10

    # ── Clinical scoring thresholds (computed from train in train.py) ─────────
    # Stored here as fallback defaults; actual values are saved to JSON after
    # first training run and reloaded from there for inference.
    WT_QUANTILE: float = 0.75
    TC_QUANTILE: float = 0.75
    ET_QUANTILE: float = 0.75

    # ── Risk label boundaries ─────────────────────────────────────────────────
    # score <= LOW_MAX  → Low Risk  (0)
    # score <= MED_MAX  → Med Risk  (1)
    # score >  MED_MAX  → High Risk (2)
    LOW_MAX: int = 1
    MED_MAX: int = 3

    # ── Candidate clinical columns (robust to NB03 feature drops) ─────────────
    CLINICAL_COLUMN_CANDIDATES: tuple = (
        "wt_volume", "tc_volume", "et_volume",
        "frontal", "temporal", "parietal", "occipital",
        "left", "right", "bilateral",
        "word_count", "sentence_count", "lobe_count",
        "edema", "necrosis", "ventricle", "compression",
    )

    # ── Risk class names ──────────────────────────────────────────────────────
    RISK_NAMES: tuple = ("Low Risk", "Medium Risk", "High Risk")

    @property
    def fusion_dir(self) -> Path:
        """Return the encoder-specific fusion dataset directory."""
        return self.FUSION_BASE_DIR / self.TEXT_ENCODER

    @property
    def nlp_dir(self) -> Path:
        """Return the NLP embeddings directory for the chosen encoder."""
        return (
            self.BIOBERT_NLP_DIR
            if self.TEXT_ENCODER == "biobert"
            else self.CLINICALBERT_NLP_DIR
        )
