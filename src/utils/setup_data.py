from pathlib import Path
from src.utils.dataset_config import (
    BRATS2020_RAW_DIR,
    TEXTBRATS_RAW_DIR,
    CV_PROCESSED_DIR,
    NLP_PROCESSED_DIR,
    FUSION_PROCESSED_DIR,
    REQUIRED_DIRS,
)

def print_header():
    print("=" * 70)
    print("CortexAI Dataset Setup Check")
    print("=" * 70)

def ensure_directories():
    """
    Create required directories if they don't exist.
    """
    print("\n[1] Ensuring required directory structure...")
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {directory}")

def count_files(directory: Path):
    """
    Count real files inside a directory recursively, ignoring .gitkeep.
    """
    if not directory.exists():
        return 0

    count = 0
    for file in directory.rglob("*"):
        if file.is_file() and file.name != ".gitkeep":
            count += 1
    return count

def check_raw_data():
    """
    Check if raw datasets are present.
    """
    print("\n[2] Checking raw datasets...")

    brats_count = count_files(BRATS2020_RAW_DIR)
    textbrats_count = count_files(TEXTBRATS_RAW_DIR)

    if brats_count > 0:
        print(f"[OK] BraTS2020 raw data found: {brats_count} files")
    else:
        print("[MISSING] No BraTS2020 files found.")
        print(f"         Please place BraTS data inside: {BRATS2020_RAW_DIR}")

    if textbrats_count > 0:
        print(f"[OK] TextBraTS raw data found: {textbrats_count} files")
    else:
        print("[MISSING] No TextBraTS files found.")
        print(f"         Please place TextBraTS data inside: {TEXTBRATS_RAW_DIR}")

def check_processed_dirs():
    """
    Check processed folders.
    """
    print("\n[3] Checking processed data directories...")

    processed_dirs = {
        "CV processed": CV_PROCESSED_DIR,
        "NLP processed": NLP_PROCESSED_DIR,
        "Fusion processed": FUSION_PROCESSED_DIR,
    }

    for name, path in processed_dirs.items():
        if path.exists():
            print(f"[OK] {name} directory ready: {path}")
        else:
            print(f"[MISSING] {name} directory missing: {path}")

def print_next_steps():
    print("\n[4] Next steps for team members:")
    print("  1. Download BraTS2020 dataset and place it inside datasets/raw/brats2020/")
    print("  2. Download TextBraTS dataset and place it inside datasets/raw/textbrats/")
    print("  3. Re-run this script to verify dataset availability.")
    print("  4. Start module-specific preprocessing/training scripts.")

def main():
    print_header()
    ensure_directories()
    check_raw_data()
    check_processed_dirs()
    print_next_steps()
    print("\nSetup check completed.")

if __name__ == "__main__":
    main()
