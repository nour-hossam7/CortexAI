from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================
# DATASET ROOTS
# =========================
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
SAMPLE_DATA_DIR = DATASETS_DIR / "sample_data"

# =========================
# RAW DATA PATHS
# =========================
BRATS2020_RAW_DIR = RAW_DATA_DIR / "brats2020"
TEXTBRATS_RAW_DIR = RAW_DATA_DIR / "textbrats"

# =========================
# PROCESSED DATA PATHS
# =========================
CV_PROCESSED_DIR = PROCESSED_DATA_DIR / "cv"
NLP_PROCESSED_DIR = PROCESSED_DATA_DIR / "nlp"
FUSION_PROCESSED_DIR = PROCESSED_DATA_DIR / "fusion"

# =========================
# OPTIONAL SUBFOLDERS (can be used later)
# =========================
CV_SPLITS_DIR = CV_PROCESSED_DIR / "splits"
CV_MASKS_DIR = CV_PROCESSED_DIR / "masks"

NLP_CLEAN_DIR = NLP_PROCESSED_DIR / "clean_text"
NLP_EMBEDDINGS_DIR = NLP_PROCESSED_DIR / "embeddings"

FUSION_TABLES_DIR = FUSION_PROCESSED_DIR / "tables"
FUSION_MAPPINGS_DIR = FUSION_PROCESSED_DIR / "mappings"

# =========================
# EXPECTED RAW FILE EXTENSIONS
# =========================
BRATS_EXTENSIONS = [".nii", ".nii.gz"]
TEXT_EXTENSIONS = [".txt", ".npy", ".json", ".csv"]

# =========================
# HELPER LISTS
# =========================
REQUIRED_DIRS = [
    DATASETS_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    BRATS2020_RAW_DIR,
    TEXTBRATS_RAW_DIR,
    CV_PROCESSED_DIR,
    NLP_PROCESSED_DIR,
    FUSION_PROCESSED_DIR,
]
