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

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from .config import FusionConfig
from .dataset import get_fusion_dataloaders
from .fusion_model import FusionModel, build_fusion_model

__all__ = ["train_fusion"]


# ── Metrics ───────────────────────────────────────────────────────────────────

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    epoch:      int,
    model:      FusionModel,
    optimizer:  torch.optim.Optimizer,
    scaler:     GradScaler,
    best_acc:   float,
    config:     FusionConfig,
) -> None:
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CHECKPOINT_DIR / "best_fusion.pth"
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
    model:     FusionModel,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    config:    FusionConfig,
) -> tuple[int, float]:
    path = config.CHECKPOINT_DIR / "best_fusion.pth"
    if not path.exists():
        return 0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    print(f"  Resumed from epoch {ckpt['epoch']} | Best Acc: {ckpt['best_acc']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_acc"]


# ── One epoch ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:         FusionModel,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    loss_fn:       nn.Module,
    scaler:        GradScaler,
    device:        torch.device,
) -> tuple[float, float]:
    """Returns (mean_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for batch in loader:
        img  = batch["image_features"].to(device)
        txt  = batch["text_features"].to(device)
        lbl  = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            logits = model(img, txt)
            loss   = loss_fn(logits, lbl)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc  += _accuracy(logits, lbl)
        n_batches  += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def validate_one_epoch(
    model:   FusionModel,
    loader:  DataLoader,
    loss_fn: nn.Module,
    device:  torch.device,
) -> tuple[float, float]:
    """Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            img  = batch["image_features"].to(device)
            txt  = batch["text_features"].to(device)
            lbl  = batch["label"].to(device)

            logits = model(img, txt)
            loss   = loss_fn(logits, lbl)

            total_loss += loss.item()
            total_acc  += _accuracy(logits, lbl)
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


# ── Main train loop ───────────────────────────────────────────────────────────

def train_fusion(config: FusionConfig | None = None) -> None:
    """
    Full training pipeline for the fusion module.

    Steps:
        1. Build model, optimizer, loss, dataloaders
        2. Resume from checkpoint if available
        3. Train + validate for NUM_EPOCHS
        4. Save best checkpoint by validation accuracy
        5. Log epoch results and save history JSON

    Output
    ------
    Best model saved to:
        models/fusion/best_fusion.pth
    Training history saved to:
        models/fusion/training_history.json
    """
    cfg = config or FusionConfig()

    torch.manual_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading fusion features...")
    train_loader, val_loader, _ = get_fusion_dataloaders(cfg)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model   = build_fusion_model(cfg).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scaler = GradScaler(device.type)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_acc = load_checkpoint(model, optimizer, scaler, device, cfg)

    # ── Training loop ─────────────────────────────────────────────────────────
    history: list[dict] = []

    print(f"\nStarting fusion training from epoch {start_epoch + 1} / {cfg.NUM_EPOCHS}")
    print("-" * 72)

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, loss_fn, device
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

    # ── Save history ──────────────────────────────────────────────────────────
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = cfg.CHECKPOINT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {history_path}")


def main() -> None:
    from .config import FusionConfig
    train_fusion(FusionConfig())


if __name__ == "__main__":
    main()
