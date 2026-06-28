"""
Training pipeline for BraTS2020 SegResNet.

Handles:

- Training
- Validation
- Checkpointing
- Metric logging

Author:
Mariam Mohamed
"""

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.amp import GradScaler

from monai.inferers import SlidingWindowInferer

from .config import Config
from .model import build_model
from .losses import build_loss
from .metrics import build_metric, get_post_transforms
from .dataloader import get_train_dataloader, get_validation_dataloader

__all__ = [
    "train",
]


# ── Device ────────────────────────────────────────────────────────────────────

def set_device() -> torch.device:
    """
    Select training device.

    Returns cuda if available, otherwise cpu.
    Designed as a function to allow easy extension
    to mps or multi-GPU in the future.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    best_dice: float,
) -> None:
    """
    Save model checkpoint to Config.CHECKPOINT_DIR.

    Saves:
        - model state dict
        - optimizer state dict
        - scaler state dict
        - epoch number
        - best validation Dice score

    Parameters
    ----------
    epoch : int
        Current epoch number.
    model : nn.Module
        Trained model.
    optimizer : Optimizer
        Current optimizer state.
    scaler : GradScaler
        Current AMP scaler state.
    best_dice : float
        Best mean Dice score achieved so far.
    """

    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Config.CHECKPOINT_DIR / "best_model.pth"

    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "best_dice":       best_dice,
        },
        checkpoint_path,
    )

    print(f"  Checkpoint saved → {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    """
    Load checkpoint from Config.CHECKPOINT_DIR if it exists.

    Parameters
    ----------
    model : nn.Module
    optimizer : Optimizer
    scaler : GradScaler
    device : torch.device

    Returns
    -------
    start_epoch : int
        Epoch to resume from (0 if no checkpoint found).
    best_dice : float
        Best Dice score from previous run (0.0 if no checkpoint found).
    """

    checkpoint_path = Config.CHECKPOINT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        return 0, 0.0

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_dice   = checkpoint["best_dice"]

    print(f"  Resumed from epoch {checkpoint['epoch']} | Best Dice: {best_dice:.4f}")

    return start_epoch, best_dice


# ── Train One Epoch ───────────────────────────────────────────────────────────

def train_one_epoch(
    model:         torch.nn.Module,
    loader:        torch.utils.data.DataLoader,
    optimizer:     torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    scaler:        GradScaler,
    device:        torch.device,
) -> float:
    """
    Run one full training epoch.

    Each .pt sample yields NUM_SAMPLES patches (RandCropByPosNegLabeld),
    and the train DataLoader is built with collate_fn=list_data_collate
    (see dataloader.py), which collates all patches across the whole
    batch into a single dict — image: (B * NUM_SAMPLES, 4, 128, 128, 128).
    So `batch` here is always a dict in practice. The isinstance check
    below is kept defensively in case the DataLoader is ever built
    without that collate_fn (it would then yield a list of patch dicts
    instead), but the for-loop normally runs exactly once per batch.

    Uses mixed precision (autocast + GradScaler) to reduce VRAM usage
    by ~40% with negligible accuracy impact.

    Parameters
    ----------
    model         : SegResNet in train() mode
    loader        : Training DataLoader
    optimizer     : AdamW
    loss_function : DiceCELoss
    scaler        : GradScaler for AMP
    device        : cuda or cpu

    Returns
    -------
    float
        Mean training loss over all patches in the epoch.
    """

    model.train()

    epoch_loss  = 0.0
    patch_count = 0

    for batch in loader:

        # Defensive normalization — see docstring above.
        if isinstance(batch, dict):
            patches = [batch]
        else:
            patches = batch

        for patch in patches:

            images = patch["image"].to(device)   # (B, 4, 128, 128, 128)
            labels = patch["label"].to(device)   # (B, 1, 128, 128, 128)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(images)           # (B, 4, 128, 128, 128)
                loss   = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            patch_count += 1

    return epoch_loss / max(patch_count, 1)


# ── Validate One Epoch ────────────────────────────────────────────────────────

def validate_one_epoch(
    model:          torch.nn.Module,
    loader:         torch.utils.data.DataLoader,
    inferer:        SlidingWindowInferer,
    metric:         object,
    post_pred:      object,
    post_label:     object,
    device:         torch.device,
) -> float:
    """
    Run one full validation epoch using sliding window inference.

    Sliding window inference is used instead of patch sampling —
    the full cropped volume is passed to the inferer, which tiles
    it into overlapping patches, runs the model on each, and
    reconstructs the full prediction.

    This gives a more accurate evaluation than patch-based inference.

    Parameters
    ----------
    model      : SegResNet in eval() mode
    loader     : Validation DataLoader (batch_size=1, full volumes)
    inferer    : SlidingWindowInferer
    metric     : DiceMetric
    post_pred  : post-processing for predictions (softmax + argmax + one-hot)
    post_label : post-processing for labels (one-hot)
    device     : cuda or cpu

    Returns
    -------
    float
        Mean Dice score across all tumor classes (NCR, Edema, ET).
        Background is excluded (Config.INCLUDE_BACKGROUND = False).
    """

    model.eval()
    metric.reset()

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)   # (1, 4, H, W, D)
            labels = batch["label"].to(device)   # (1, 1, H, W, D)

            # sliding window inference over full volume
            logits = inferer(images, model)      # (1, 4, H, W, D)

            # post-process
            pred  = post_pred(logits)            # (1, 4, H, W, D) one-hot
            label = post_label(labels)           # (1, 4, H, W, D) one-hot

            metric(y_pred=pred, y=label)

    # aggregate mean Dice across all validation volumes
    dice_scores = metric.aggregate()             # (num_classes,) or scalar

    mean_dice = dice_scores.mean().item()

    metric.reset()

    return mean_dice


# ── Main Train Loop ───────────────────────────────────────────────────────────

def train() -> None:
    """
    Full training pipeline for BraTS2020 SegResNet.

    Steps:
        1. Set device
        2. Build model, loss, metric, dataloaders, optimizer, inferer, scaler
        3. Resume from checkpoint if available
        4. Run train + validate loop for NUM_EPOCHS
        5. Save best checkpoint based on validation Dice
        6. Log epoch results

    Output
    ------
    Best model saved to:
        Config.CHECKPOINT_DIR / best_model.pth

    Console logs per epoch:
        Epoch | Train Loss | Val Dice | (saved marker)
    """

    # ── Setup ─────────────────────────────────────────────────────────────────

    device = set_device()
    print(f"Device: {device}")

    model         = build_model().to(device)
    loss_function = build_loss()
    metric        = build_metric()

    post_pred, post_label = get_post_transforms()

    train_loader      = get_train_dataloader()
    validation_loader = get_validation_dataloader()

    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scaler = GradScaler(device.type)

    inferer = SlidingWindowInferer(
        roi_size=Config.ROI_SIZE,
        sw_batch_size=1,
        overlap=0.5,
    )

    # ── Resume ────────────────────────────────────────────────────────────────

    start_epoch, best_dice = load_checkpoint(
        model, optimizer, scaler, device
    )

    # ── Training Loop ─────────────────────────────────────────────────────────

    print(f"\nStarting training from epoch {start_epoch + 1} / {Config.NUM_EPOCHS}")
    print("-" * 65)

    for epoch in range(start_epoch, Config.NUM_EPOCHS):

        # train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            scaler=scaler,
            device=device,
        )

        # validate
        val_dice = validate_one_epoch(
            model=model,
            loader=validation_loader,
            inferer=inferer,
            metric=metric,
            post_pred=post_pred,
            post_label=post_label,
            device=device,
        )

        # checkpoint
        saved_marker = ""

        if val_dice > best_dice:
            best_dice    = val_dice
            saved_marker = " ← saved"

            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_dice=best_dice,
            )

        # log
        print(
            f"Epoch {epoch + 1:03d}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Dice:   {val_dice:.4f}"
            f"{saved_marker}"
        )

    print("-" * 65)
    print(f"Training complete. Best Val Dice: {best_dice:.4f}")
    print(f"Checkpoint: {Config.CHECKPOINT_DIR / 'best_model.pth'}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point when running train.py directly.
    """

    from .preprocessing import set_seed
    set_seed()
    train()


if __name__ == "__main__":
    main()
