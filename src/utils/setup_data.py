from src.utils.dataset_config import (
    REQUIRED_DIRS,
    BRATS_RAW_DIR,
    TEXTBRATS_RAW_DIR,
    CV_PROCESSED_DIR,
    NLP_PROCESSED_DIR,
    FUSION_PROCESSED_DIR,
)

def create_directories():
    print("Checking required dataset directories...\n")
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {directory}")

def check_dataset_contents():
    print("\nChecking raw dataset availability...\n")

    brats_files = list(BRATS_RAW_DIR.glob("*"))
    textbrats_files = list(TEXTBRATS_RAW_DIR.glob("*"))

    if brats_files:
        print(f"[FOUND] BraTS2020 dataset detected in: {BRATS_RAW_DIR}")
    else:
        print(f"[MISSING] No files found in: {BRATS_RAW_DIR}")

    if textbrats_files:
        print(f"[FOUND] TextBraTS dataset detected in: {TEXTBRATS_RAW_DIR}")
    else:
        print(f"[MISSING] No files found in: {TEXTBRATS_RAW_DIR}")

    print("\nProcessed dataset directories:")
    print(f"[READY] CV processed folder: {CV_PROCESSED_DIR}")
    print(f"[READY] NLP processed folder: {NLP_PROCESSED_DIR}")
    print(f"[READY] Fusion processed folder: {FUSION_PROCESSED_DIR}")

if __name__ == "__main__":
    create_directories()
    check_dataset_contents()
