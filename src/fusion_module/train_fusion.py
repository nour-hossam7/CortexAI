"""
Training pipeline for the CortexAI Fusion Module.

Handles:
    - Training loop with mixed precision (AMP)
    - Validation loop
    - Checkpoint saving (best model by val accuracy)
    - Metric logging

Author:
    Ammar Kamal
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from .config import FusionConfig
from .dataset import FusionDataset, load_fusion_split
from .fusion_model import ClinicalDecisionModel, build_model
from .train import assign_labels, compute_thresholds, get_clinical_columns, make_loader, scale_clinical

__all__ = ["train_fusion"]

SPLITS = ("train", "validation", "test")


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def save_checkpoint(
    epoch:      int,
    model:      ClinicalDecisionModel,
    optimizer:  torch.optim.Optimizer,
    scaler:     GradScaler,
    best_acc:   float,
    config:     FusionConfig,
) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODEL_DIR / "best_fusion.pth"
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "best_acc":        best_acc,
        },
        path,
    )
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(
    model:     ClinicalDecisionModel,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    config:    FusionConfig,
) -> tuple[int, float]:
    path = config.MODEL_DIR / "best_fusion.pth"
    if not path.exists():
        return 0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    print(f"  Resumed from epoch {ckpt['epoch']} | Best Acc: {ckpt['best_acc']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_acc"]


def train_one_epoch(
    model:         ClinicalDecisionModel,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    loss_fn:       nn.Module,
    scaler:        GradScaler,
    device:        torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for batch in loader:
        img  = batch["image"].to(device)
        txt  = batch["text"].to(device)
        clin = batch["clinical"].to(device)
        lbl  = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(img, txt, clin)
            loss   = loss_fn(logits, lbl)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc  += _accuracy(logits, lbl)
        n_batches  += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def validate_one_epoch(
    model:   ClinicalDecisionModel,
    loader:  DataLoader,
    loss_fn: nn.Module,
    device:  torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            img  = batch["image"].to(device)
            txt  = batch["text"].to(device)
            clin = batch["clinical"].to(device)
            lbl  = batch["label"].to(device)

            logits, _ = model(img, txt, clin)
            loss   = loss_fn(logits, lbl)

            total_loss += loss.item()
            total_acc  += _accuracy(logits, lbl)
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train_fusion(config: FusionConfig | None = None) -> None:
    cfg = config or FusionConfig()

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPR_DIR.mkdir(parents=True, exist_ok=True)

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

    clinical_dim = len(clinical_columns)
    model   = build_model(clinical_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ClinicalDecisionModel  trainable params: {total_params:,}")

    train_labels  = splits_data["train"]["clinical"]["risk_label"].values
    class_counts  = np.bincount(train_labels)
    class_weights = torch.tensor(
        len(train_labels) / (len(class_counts) * class_counts),
        dtype=torch.float32,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scaler = GradScaler(device.type)

    start_epoch, best_acc = load_checkpoint(model, optimizer, scaler, device, cfg)

    history: list[dict] = []

    print(f"\nStarting fusion training from epoch {start_epoch + 1} / {cfg.NUM_EPOCHS}")
    print("-" * 72)

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, loss_fn, scaler, device
        )

        val_loss, val_acc = validate_one_epoch(
            model, loaders["validation"], loss_fn, device
        )

        saved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            saved    = " ← saved"
            save_checkpoint(epoch, model, optimizer, scaler, best_acc, cfg)

        print(
            f"Epoch {epoch + 1:03d}/{cfg.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            f"{saved}"
        )

        history.append({
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
        })

    print("-" * 72)
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history_path = cfg.MODEL_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {history_path}")


def main() -> None:
    from .config import FusionConfig
    train_fusion(FusionConfig())


if __name__ == "__main__":
    main()
