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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler

from monai.inferers import SlidingWindowInferer
from monai.data import decollate_batch

from .config import Config
from .model import build_model
from .losses import build_loss
from .metrics import build_metric, get_post_transforms
from .dataloader import get_train_dataloader, get_validation_dataloader

__all__ = [
    "train",
    "save_checkpoint",
    "save_last_checkpoint",
    "load_checkpoint",
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
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    best_dice: float,
) -> None:
    """
    Save the BEST model checkpoint to Config.CHECKPOINT_DIR / "best_model.pth".

    Only called when validation Dice improves — see save_last_checkpoint()
    below for the unconditional per-epoch save used to survive crashes
    during a non-improving streak.

    Saves:
        - model state dict
        - optimizer state dict
        - scheduler state dict
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
    scheduler : LRScheduler
        Current CosineAnnealingLR scheduler state.
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
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "best_dice":       best_dice,
        },
        checkpoint_path,
    )

    print(f"  Checkpoint saved → {checkpoint_path}")


def save_last_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    best_dice: float,
) -> None:
    """
    Save the LATEST checkpoint to Config.CHECKPOINT_DIR / "last_model.pth",
    unconditionally, every epoch — regardless of whether validation Dice
    improved.

    🔧 FIX: save_checkpoint() above only runs inside `if val_dice >
    best_dice`. With patience=20, a run that crashes (Kaggle session
    timeout, disconnect, OOM) during a non-improving streak loses up to
    20 epochs of progress, since nothing was written to disk since the
    last improvement. Notebook 08 avoided this by writing both
    best_model.pth (on improvement) and last_model.pth (every epoch);
    this restores that behavior for train.py.

    To resume from the latest epoch instead of the best one, pass this
    file's path to load_checkpoint() / load_model() explicitly.

    Parameters
    ----------
    Same as save_checkpoint().
    """

    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Config.CHECKPOINT_DIR / "last_model.pth"

    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "best_dice":       best_dice,
        },
        checkpoint_path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> tuple[int, float]:
    """
    Load checkpoint from Config.CHECKPOINT_DIR if it exists.

    Handles two checkpoint shapes, since both exist for this project:

    - Full training checkpoint, saved by THIS module's save_checkpoint()
      or save_last_checkpoint(): a dict with "model_state",
      "optimizer_state", "scheduler_state", "scaler_state", "epoch",
      "best_dice". Resumes training exactly where it left off —
      optimizer momentum, LR schedule position, and AMP scaler state
      are all restored.

    - Raw state_dict, saved by the training notebooks (e.g. Notebook 08:
      `torch.save(model.state_dict(), "best_model.pth")`). This is the
      actual format of best_model.pth on disk today. It contains only
      weights — no optimizer/scheduler/epoch/best_dice were ever saved
      alongside it, so none of that can be recovered. In this case we
      load the weights and resume from epoch 0 with a fresh optimizer
      and scheduler, rather than raising KeyError trying to read state
      that was never written.

    Parameters
    ----------
    model : nn.Module
    optimizer : Optimizer
    scheduler : LRScheduler
    scaler : GradScaler
    device : torch.device
    checkpoint_path : Path | None
        Which file to resume from. Defaults to
        Config.CHECKPOINT_DIR / "best_model.pth". Pass
        Config.CHECKPOINT_DIR / "last_model.pth" instead to resume from
        the most recent epoch rather than the best-Dice epoch — useful
        after a crash mid-streak (see save_last_checkpoint()).

    Returns
    -------
    start_epoch : int
        Epoch to resume from (0 if no checkpoint, or if checkpoint is
        a raw state_dict with no recorded epoch).
    best_dice : float
        Best Dice score from previous run (0.0 if no checkpoint, or if
        checkpoint is a raw state_dict with no recorded best_dice —
        the first validation pass of this run will set the real value).
    """

    if checkpoint_path is None:
        checkpoint_path = Config.CHECKPOINT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        return 0, 0.0

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    is_full_checkpoint = isinstance(checkpoint, dict) and "model_state" in checkpoint

    if is_full_checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint["epoch"] + 1
        best_dice   = checkpoint["best_dice"]

        print(f"  Resumed full checkpoint from epoch {checkpoint['epoch']} | Best Dice: {best_dice:.4f}")

    else:
        # Raw state_dict — weights only, no training state to resume.
        model.load_state_dict(checkpoint)

        start_epoch = 0
        best_dice   = 0.0

        print(
            "  Loaded weights-only checkpoint (raw state_dict) — "
            "no optimizer/scheduler/epoch was saved alongside it. "
            "Starting from epoch 0 with a fresh optimizer/scheduler. "
            "Training will still benefit from these pretrained weights; "
            "the first validation pass will set the true best_dice."
        )

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
    loss_function:  torch.nn.Module,
    metric:         object,
    post_pred:      object,
    post_label:     object,
    device:         torch.device,
) -> tuple[float, float]:
    """
    Run one full validation epoch using sliding window inference.

    Sliding window inference is used instead of patch sampling —
    the full cropped volume is passed to the inferer, which tiles
    it into overlapping patches, runs the model on each, and
    reconstructs the full prediction.

    This gives a more accurate evaluation than patch-based inference.

    Post-processing (Activations / AsDiscrete) is applied per-sample
    via decollate_batch, not on the raw batched tensor. MONAI's
    Compose post-transforms expect a single (C, H, W, D) sample —
    feeding them a batched (B, C, H, W, D) tensor directly silently
    produces incorrect one-hot/argmax results. This matches the
    exact pattern validated end-to-end in Notebook 08
    (Advanced Training).

    Parameters
    ----------
    model         : SegResNet in eval() mode
    loader        : Validation DataLoader (batch_size=1, full volumes)
    inferer       : SlidingWindowInferer
    loss_function : DiceCELoss — used to track validation loss alongside Dice
    metric        : DiceMetric
    post_pred     : post-processing for predictions (softmax + argmax + one-hot)
    post_label    : post-processing for labels (one-hot)
    device        : cuda or cpu

    Returns
    -------
    tuple[float, float]
        (mean_loss, mean_dice) over the validation set.
        mean_dice excludes background (Config.INCLUDE_BACKGROUND = False).
    """

    model.eval()
    metric.reset()

    running_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)   # (1, 4, H, W, D)
            labels = batch["label"].to(device)   # (1, 1, H, W, D)

            # sliding window inference over full volume
            logits = inferer(images, model)      # (1, 4, H, W, D)

            loss = loss_function(logits, labels)
            running_loss += loss.item()

            # post-process per-sample, not on the raw batched tensor
            pred  = [post_pred(p)  for p in decollate_batch(logits)]
            label = [post_label(l) for l in decollate_batch(labels)]

            metric(y_pred=pred, y=label)

    # aggregate mean Dice across all validation volumes
    dice_scores = metric.aggregate()             # (num_classes,) or scalar

    mean_dice = dice_scores.mean().item()
    mean_loss = running_loss / max(len(loader), 1)

    metric.reset()

    return mean_loss, mean_dice


# ── Main Train Loop ───────────────────────────────────────────────────────────

def train() -> None:
    """
    Full training pipeline for BraTS2020 SegResNet.

    Steps:
        1. Set device
        2. Build model, loss, metric, dataloaders, optimizer, scheduler, inferer, scaler
        3. Resume from checkpoint if available
        4. Run train + validate loop for NUM_EPOCHS, with early stopping
        5. Save best checkpoint based on validation Dice
        6. Log epoch results

    Matches the training configuration validated end-to-end in
    Notebook 08 (Advanced Training): AdamW + CosineAnnealingLR
    (T_max=NUM_EPOCHS), early stopping with patience=20 epochs of no
    Dice improvement.

    Output
    ------
    Best model saved to:
        Config.CHECKPOINT_DIR / best_model.pth

    Console logs per epoch:
        Epoch | Train Loss | Val Loss | Val Dice | LR | (saved marker)
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

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS,
    )

    scaler = GradScaler(device.type)

    inferer = SlidingWindowInferer(
        roi_size=Config.ROI_SIZE,
        sw_batch_size=1,
        overlap=0.5,
    )

    # ── Resume ────────────────────────────────────────────────────────────────

    start_epoch, best_dice = load_checkpoint(
        model, optimizer, scheduler, scaler, device
    )

    # ── Early Stopping ────────────────────────────────────────────────────────

    patience                   = 20
    epochs_without_improvement = 0

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
        val_loss, val_dice = validate_one_epoch(
            model=model,
            loader=validation_loader,
            inferer=inferer,
            loss_function=loss_function,
            metric=metric,
            post_pred=post_pred,
            post_label=post_label,
            device=device,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # checkpoint
        saved_marker = ""

        if val_dice > best_dice:
            best_dice    = val_dice
            saved_marker = " ← saved"
            epochs_without_improvement = 0

            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_dice=best_dice,
            )

        else:
            epochs_without_improvement += 1

        # 🔧 FIX: unconditional per-epoch save, independent of best_dice
        # improvement — see save_last_checkpoint() docstring.
        save_last_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_dice=best_dice,
        )

        # log
        print(
            f"Epoch {epoch + 1:03d}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.2e}"
            f"{saved_marker}"
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered (no improvement for {patience} epochs).")
            break

    print("-" * 65)
    print(f"Training complete. Best Val Dice: {best_dice:.4f}")
    print(f"Best checkpoint: {Config.CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"Last checkpoint: {Config.CHECKPOINT_DIR / 'last_model.pth'}")


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
