"""
Fusion Data Preparation — NB01 as a runnable script.

Loads CV bottleneck features + NLP embeddings, verifies patient alignment,
sorts both arrays by patient_id, and saves {split}_fusion.npz files.

Usage
-----
    python -m fusion_module.data_preparation

    # with custom paths:
    python -m fusion_module.data_preparation \
        --cv-results-dir  reports/results \
        --text-encoder    biobert \
        --output-dir      datasets/processed/fusion

Author: Ammar Kamal
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import FusionConfig
from .dataset import (
    build_fusion_dataset,
    load_cv_features,
    load_nlp_embeddings,
    save_fusion_dataset,
    verify_alignment,
)


def prepare_fusion_data(config: FusionConfig | None = None) -> None:
    """Run the full NB01 data preparation pipeline."""
    cfg = config or FusionConfig()

    print("=" * 70)
    print("Fusion Data Preparation")
    print("=" * 70)
    print(f"  Text encoder : {cfg.TEXT_ENCODER}")
    print(f"  CV dir       : {cfg.CV_RESULTS_DIR}")
    print(f"  NLP dir      : {cfg.nlp_dir}")
    print(f"  Output dir   : {cfg.fusion_dir}")

    # Load
    print("\nLoading CV features...")
    cv = load_cv_features(cfg.CV_RESULTS_DIR)

    print("Loading NLP embeddings...")
    nlp = load_nlp_embeddings(cfg.nlp_dir)

    # Align
    verify_alignment(cv, nlp, model_name=cfg.TEXT_ENCODER)

    # Build
    print("\nBuilding fusion dataset...")
    fusion = build_fusion_dataset(cv, nlp)

    # Save
    print("\nSaving...")
    save_fusion_dataset(fusion, cfg.FUSION_BASE_DIR, cfg.TEXT_ENCODER)

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CortexAI fusion dataset.")
    parser.add_argument("--text-encoder",    default="biobert", choices=["biobert", "clinicalbert"])
    parser.add_argument("--cv-results-dir",  type=Path, default=Path("reports/results"))
    parser.add_argument("--output-dir",      type=Path, default=Path("datasets/processed/fusion"))
    args = parser.parse_args()

    from dataclasses import replace
    cfg = replace(
        FusionConfig(),
        TEXT_ENCODER    = args.text_encoder,
        CV_RESULTS_DIR  = args.cv_results_dir,
        FUSION_BASE_DIR = args.output_dir,
    )
    prepare_fusion_data(cfg)


if __name__ == "__main__":
    main()
