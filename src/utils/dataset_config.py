from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset root
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Raw datasets
RAW_DIR = DATASETS_DIR / "raw"
BRATS_RAW_DIR = RAW_DIR / "brats2020"
TEXTBRATS_RAW_DIR = RAW_DIR / "textbrats"

# Processed datasets
PROCESSED_DIR = DATASETS_DIR / "processed"
CV_PROCESSED_DIR = PROCESSED_DIR / "cv"
NLP_PROCESSED_DIR = PROCESSED_DIR / "nlp"
FUSION_PROCESSED_DIR = PROCESSED_DIR / "fusion"

# Sample data
SAMPLE_DATA_DIR = DATASETS_DIR / "sample_data"

REQUIRED_DIRS = [
    DATASETS_DIR,
    RAW_DIR,
    BRATS_RAW_DIR,
    TEXTBRATS_RAW_DIR,
    PROCESSED_DIR,
    CV_PROCESSED_DIR,
    NLP_PROCESSED_DIR,
    FUSION_PROCESSED_DIR,
    SAMPLE_DATA_DIR,
]