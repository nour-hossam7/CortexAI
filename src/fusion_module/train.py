"""
Training pipeline for the CortexAI Fusion Module.

Implements NB04 logic as a runnable script and importable function.

Key design decisions (all matching NB04):
  - Rule-based labels computed from TRAIN statistics only (no leakage)
  - StandardScaler fitted on train, applied to val/test
  - Thresholds + scaler saved for reproducible inference
  - Class-weighted CrossEntropyLoss for imbalanced risk labels
  - CosineAnnealingLR scheduler
  - Early stopping on validation accuracy
  - Best checkpoint + last checkpoint both saved
  - Unified representations extracted and saved after training

Usage
-----
    python -m fusion_module.train

    # or with custom paths:
    python -m fusion_module.train \
        --text-encoder biobert \
        --fusion-dir  datasets/processed/fusion \
        --clinical-dir datasets/processed/clinical_features \
        --model-dir   models/fusion \
        --epochs 50

Author: Ammar Kamal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import FusionConfig
from .dataset import FusionDataset, load_fusion_split
from .fusion_model import ClinicalDecisionModel, build_model

__all__ = ["train_fusion"]

SPLITS = ("train", "validation", "test")


# ── Label generation ─────────────────────────────────────────────────────────

def compute_thresholds(train_clinical: pd.DataFrame, config: FusionConfig) -> dict:
    """Compute volume thresholds from train quantiles — no leakage."""
    return {
        "WT_THRESHOLD": float(train_clinical["wt_volume"].quantile(config.WT_QUANTILE)),
        "TC_THRESHOLD": float(train_clinical["tc_volume"].quantile(config.TC_QUANTILE)),
        "ET_THRESHOLD": float(train_clinical["et_volume"].quantile(config.ET_QUANTILE)),
    }


def compute_clinical_score(df: pd.DataFrame, thresholds: dict) -> list[int]:
    """
    Rule-based severity score per patient.

    Scoring rubric (matches NB04 code, not the markdown):
        WT Volume ≥ threshold  → +2
        ET Volume ≥ threshold  → +2
        TC Volume ≥ threshold  → +1
        Lobe count ≥ 3         → +1
        Bilateral              → +1
        Max possible score     =  7
    """
    wt = thresholds["WT_THRESHOLD"]
    tc = thresholds["TC_THRESHOLD"]
    et = thresholds["ET_THRESHOLD"]

    scores = []
    for _, row in df.iterrows():
        score = 0
        if row["wt_volume"] >= wt: score += 2
        if row["et_volume"] >= et: score += 2
        if row["tc_volume"] >= tc: score += 1
        if row.get("lobe_count", 0) >= 3: score += 1
        if row.get("bilateral", 0) == 1:  score += 1
        scores.append(score)
    return scores


def risk_label(score: int, config: FusionConfig) -> int:
    if score <= config.LOW_MAX: return 0
    if score <= config.MED_MAX: return 1
    return 2


def assign_labels(
    clinical: pd.DataFrame,
    thresholds: dict,
    config: FusionConfig,
) -> pd.DataFrame:
    """Add severity_score and risk_label columns."""
    clinical = clinical.copy()
    clinical["severity_score"] = compute_clinical_score(clinical, thresholds)
    clinical["risk_label"]     = clinical["severity_score"].apply(lambda s: risk_label(s, config))
    return clinical


# ── Scaling ───────────────────────────────────────────────────────────────────

def get_clinical_columns(
    train_clinical: pd.DataFrame,
    candidates: tuple[str, ...],
) -> list[str]:
    """Return candidate columns that actually exist in the dataframe."""
    return [c for c in candidates if c in train_clinical.columns]


def scale_clinical(
    train_clinical:      pd.DataFrame,
    validation_clinical: pd.DataFrame,
    test_clinical:       pd.DataFrame,
    clinical_columns:    list[str],
    model_dir:           Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train, transform all splits, save scaler."""
    scaler = StandardScaler()
    train_clinical      = train_clinical.copy()
    validation_clinical = validation_clinical.copy()
    test_clinical       = test_clinical.copy()

    train_clinical[clinical_columns]      = scaler.fit_transform(train_clinical[clinical_columns])
    validation_clinical[clinical_columns] = scaler.transform(validation_clinical[clinical_columns])
    test_clinical[clinical_columns]       = scaler.transform(test_clinical[clinical_columns])

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "clinical_scaler.pkl")
    print(f"Scaler saved → {model_dir / 'clinical_scaler.pkl'}")

    return train_clinical, validation_clinical, test_clinical, scaler


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_loader(
    image:    np.ndarray,
    text:     np.ndarray,
    clinical: pd.DataFrame,
    clinical_columns: list[str],
    batch_size: int,
    shuffle: bool,
    patient_ids: list[str] | None = None,
) -> DataLoader:
    ds = FusionDataset(
        image_features    = image,
        text_features     = text,
        clinical_features = clinical[clinical_columns].values,
        labels            = clinical["risk_label"].values,
        patient_ids       = patient_ids or clinical["patient_id"].tolist(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    model:     ClinicalDecisionModel,
    loader:    DataLoader,
    device:    torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    training:  bool = True,
) -> tuple[float, float]:
    """Return (mean_loss, accuracy)."""
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            image    = batch["image"].to(device)
            text     = batch["text"].to(device)
            clinical = batch["clinical"].to(device)
            labels   = batch["label"].to(device)

            logits, _ = model(image, text, clinical)
            loss      = criterion(logits, labels)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / max(n, 1), acc


# ── Representation extraction ─────────────────────────────────────────────────

def extract_representations(
    model:  ClinicalDecisionModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract unified_repr + predictions + confidences + labels for all patients."""
    model.eval()
    reprs, preds, confs, labs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            logits, unified_repr = model(
                batch["image"].to(device),
                batch["text"].to(device),
                batch["clinical"].to(device),
            )
            probs = torch.softmax(logits, dim=1)
            reprs.extend(unified_repr.cpu().numpy())
            preds.extend(probs.argmax(dim=1).cpu().numpy())
            confs.extend(probs.max(dim=1).values.cpu().numpy())
            labs.extend(batch["label"].numpy())

    return (
        np.array(reprs),
        np.array(preds),
        np.array(confs),
        np.array(labs),
    )


# ── Main training function ────────────────────────────────────────────────────

def train_fusion(config: FusionConfig | None = None) -> None:
    """
    Full training pipeline for the CortexAI fusion module.

    Steps:
        1. Load fusion features + clinical tables
        2. Compute labels (thresholds from train only)
        3. Scale clinical features (scaler fit on train only)
        4. Build DataLoaders
        5. Train with early stopping
        6. Evaluate on test set
        7. Extract and save unified representations
        8. Save training history
    """
    cfg = config or FusionConfig()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPR_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading fusion features...")
    splits_data: dict = {}
    for split in SPLITS:
        fusion   = load_fusion_split(cfg.fusion_dir, split)
        clinical = pd.read_csv(cfg.CLINICAL_DIR / f"{split}_clinical_features.csv")
        splits_data[split] = {
            "image":    fusion["image_features"],
            "text":     fusion["text_features"],
            "clinical": clinical,
        }
        print(f"  {split:12}  image {fusion['image_features'].shape}  "
              f"text {fusion['text_features'].shape}  clinical {clinical.shape}")

    # ── Labels ────────────────────────────────────────────────────────────────
    thresholds = compute_thresholds(splits_data["train"]["clinical"], cfg)
    print(f"\nThresholds (from train): {thresholds}")

    (cfg.MODEL_DIR / "severity_thresholds.json").write_text(
        json.dumps(thresholds, indent=4)
    )

    for split in SPLITS:
        splits_data[split]["clinical"] = assign_labels(
            splits_data[split]["clinical"], thresholds, cfg
        )

    for split in SPLITS:
        vc = splits_data[split]["clinical"]["risk_label"].value_counts().sort_index()
        print(f"  {split:12}  Low={vc.get(0,0)}  Medium={vc.get(1,0)}  High={vc.get(2,0)}")

    # ── Clinical columns + scaling ────────────────────────────────────────────
    clinical_columns = get_clinical_columns(
        splits_data["train"]["clinical"],
        cfg.CLINICAL_COLUMN_CANDIDATES,
    )
    print(f"\nClinical columns ({len(clinical_columns)}): {clinical_columns}")

    (
        splits_data["train"]["clinical"],
        splits_data["validation"]["clinical"],
        splits_data["test"]["clinical"],
        _,
    ) = scale_clinical(
        splits_data["train"]["clinical"],
        splits_data["validation"]["clinical"],
        splits_data["test"]["clinical"],
        clinical_columns,
        cfg.MODEL_DIR,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loaders = {
        split: make_loader(
            image    = splits_data[split]["image"],
            text     = splits_data[split]["text"],
            clinical = splits_data[split]["clinical"],
            clinical_columns = clinical_columns,
            batch_size = cfg.BATCH_SIZE,
            shuffle    = (split == "train"),
        )
        for split in SPLITS
    }

    # ── Model ─────────────────────────────────────────────────────────────────
    clinical_dim = len(clinical_columns)
    model        = build_model(clinical_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ClinicalDecisionModel  trainable params: {total_params:,}")

    # Class-weighted loss
    train_labels  = splits_data["train"]["clinical"]["risk_label"].values
    class_counts  = np.bincount(train_labels)
    class_weights = torch.tensor(
        len(train_labels) / (len(class_counts) * class_counts),
        dtype=torch.float32,
    ).to(device)
    np.save(cfg.MODEL_DIR / "class_weights.npy", class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    patience_count = 0
    history        = []

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], device, criterion, optimizer, training=True
        )
        val_loss, val_acc = run_epoch(
            model, loaders["validation"], device, criterion, training=False
        )
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  {val_loss:>8.4f}  {val_acc:>7.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":         best_val_acc,
                "clinical_dim":    clinical_dim,
                "clinical_cols":   clinical_columns,
            }, cfg.MODEL_DIR / "best_decision_model.pth")
        else:
            patience_count += 1
            if patience_count >= cfg.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"— no improvement for {cfg.PATIENCE} epochs.")
                break

        # Save last (crash recovery)
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc":         val_acc,
            "clinical_dim":    clinical_dim,
            "clinical_cols":   clinical_columns,
        }, cfg.MODEL_DIR / "last_decision_model.pth")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(cfg.MODEL_DIR / "best_decision_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            logits, _ = model(
                batch["image"].to(device),
                batch["text"].to(device),
                batch["clinical"].to(device),
            )
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    print("\n" + "=" * 55)
    print("Test Set Evaluation")
    print("=" * 55)
    print(classification_report(
        all_labels, all_preds,
        target_names=list(cfg.RISK_NAMES),
        digits=4,
    ))

    # ── Extract + save representations ────────────────────────────────────────
    for split in SPLITS:
        loader = DataLoader(
            FusionDataset(
                splits_data[split]["image"],
                splits_data[split]["text"],
                splits_data[split]["clinical"][clinical_columns].values,
                splits_data[split]["clinical"]["risk_label"].values,
                splits_data[split]["clinical"]["patient_id"].tolist(),
            ),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
        )
        reprs, preds, confs, labs = extract_representations(model, loader, device)

        np.save(cfg.REPR_DIR / f"{split}_unified_repr.npy", reprs)

        meta = splits_data[split]["clinical"][
            ["patient_id", "risk_label", "severity_score"]
        ].copy().reset_index(drop=True)
        meta["predicted_risk"] = preds
        meta["confidence"]     = confs
        meta.to_csv(cfg.REPR_DIR / f"{split}_repr_metadata.csv", index=False)

        print(f"  {split:12}  repr {reprs.shape}")

    # ── Save history ──────────────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(cfg.MODEL_DIR / "training_history.csv", index=False)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Best Val Accuracy  : {best_val_acc:.4f}")
    print(f"  Clinical columns   : {clinical_columns}")
    print(f"  Checkpoint         : {cfg.MODEL_DIR / 'best_decision_model.pth'}")
    print(f"  Representations    : {cfg.REPR_DIR}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CortexAI fusion model.")
    parser.add_argument("--text-encoder",   default="biobert", choices=["biobert", "clinicalbert"])
    parser.add_argument("--fusion-dir",     type=Path, default=Path("datasets/processed/fusion"))
    parser.add_argument("--clinical-dir",   type=Path, default=Path("datasets/processed/clinical_features"))
    parser.add_argument("--model-dir",      type=Path, default=Path("models/fusion"))
    parser.add_argument("--repr-dir",       type=Path, default=Path("reports/fusion/representations"))
    parser.add_argument("--epochs",         type=int,  default=50)
    parser.add_argument("--batch-size",     type=int,  default=32)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--patience",       type=int,  default=10)
    args = parser.parse_args()

    from dataclasses import replace
    cfg = replace(
        FusionConfig(),
        TEXT_ENCODER   = args.text_encoder,
        FUSION_BASE_DIR= args.fusion_dir,
        CLINICAL_DIR   = args.clinical_dir,
        MODEL_DIR      = args.model_dir,
        REPR_DIR       = args.repr_dir,
        NUM_EPOCHS     = args.epochs,
        BATCH_SIZE     = args.batch_size,
        LEARNING_RATE  = args.lr,
        PATIENCE       = args.patience,
    )
    train_fusion(cfg)


if __name__ == "__main__":
    main()
