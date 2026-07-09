"""
Knowledge Distillation — train a lightweight student encoder to
mimic the existing teacher (BioBERT / ClinicalBERT).

Usage
-----
    python -m src.nlp_module.train_student
    python -m src.nlp_module.train_student --epochs 20 --loss_type cosine
    python -m src.nlp_module.train_student --student_model_name distilbert-base-cased

The script is completely self-contained: it loads the teacher, builds
the student, loads cleaned reports, runs the distillation loop, and
saves checkpoints, history, and comparison plots under
``models/nlp_student/``.

Author:
Nour Hossam
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import Config as NLPConfig
from .dataset import load_clean_reports
from .distillation_config import DistillationConfig
from .distillation_utils import (
    TextDataset,
    get_loss_fn,
    compute_cosine_similarity,
    compute_mse,
    count_parameters,
    compare_models,
)
from .model import build_encoder, mean_pooling
from .student_model import build_student_encoder

__all__ = [
    "train_student",
    "evaluate",
    "load_student_checkpoint",
    "plot_training_curves",
    "save_comparison_table",
    "main",
]

logger = logging.getLogger(__name__)


# ===================================================================
# Argument parsing
# ===================================================================


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for the NLP encoder",
    )

    # Model
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default=DistillationConfig.TEACHER_MODEL_NAME,
        help=f"Teacher model (default: {DistillationConfig.TEACHER_MODEL_NAME})",
    )
    parser.add_argument(
        "--student_model_name",
        type=str,
        default=DistillationConfig.STUDENT_MODEL_NAME,
        help=f"Student model (default: {DistillationConfig.STUDENT_MODEL_NAME})",
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DistillationConfig.BATCH_SIZE,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DistillationConfig.LEARNING_RATE,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DistillationConfig.NUM_EPOCHS,
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=DistillationConfig.LOSS_TYPE,
        choices=DistillationConfig.LOSS_TYPE_CHOICES,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DistillationConfig.WEIGHT_DECAY,
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=DistillationConfig.WARMUP_STEPS,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=DistillationConfig.MAX_GRAD_NORM,
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=DistillationConfig.DEVICE,
        help="Device: 'auto', 'cuda', 'cpu'",
    )

    # Paths
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(DistillationConfig.CHECKPOINT_DIR),
    )

    # Logging
    parser.add_argument(
        "--log_interval",
        type=int,
        default=DistillationConfig.LOG_INTERVAL,
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=DistillationConfig.EVAL_INTERVAL,
    )

    return parser.parse_args(argv)


# ===================================================================
# Helper: collate function
# ===================================================================


def _collate_texts(batch: List[str]) -> List[str]:
    """Identity collate for a list of strings."""
    return batch


# ===================================================================
# Core training loop
# ===================================================================


def train_student(
    teacher_model: torch.nn.Module,
    teacher_tokenizer,
    student_model: torch.nn.Module,
    student_tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    loss_fn,
    device: torch.device,
    num_epochs: int,
    checkpoint_dir: Path,
    log_interval: int = 10,
    eval_interval: int = 50,
    max_grad_norm: float = 1.0,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    Run the distillation training loop.

    The teacher model is kept in ``eval()`` mode throughout — only
    the student model receives gradients.

    Parameters
    ----------
    teacher_model : torch.nn.Module
        Frozen teacher encoder (eval mode).
    teacher_tokenizer
        Teacher tokenizer (used for both models since vocab is shared).
    student_model : torch.nn.Module
        Trainable student encoder.
    student_tokenizer
        Student tokenizer.
    train_loader : DataLoader
        Batches of report texts for training.
    val_loader : DataLoader
        Batches of report texts for validation.
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
    loss_fn : callable
        ``(teacher_emb, student_emb) -> scalar_tensor``.
    device : torch.device
    num_epochs : int
    checkpoint_dir : Path
        Directory for saving ``best_student_model.pt`` and
        ``last_student_model.pt``.
    log_interval : int
        Log training loss every N batches.
    eval_interval : int
        Run validation every N batches.
    max_grad_norm : float
        Gradient clipping norm.

    Returns
    -------
    tuple
        ``(history, best_metrics)`` where *history* is
        ``{"train_loss": [...], "val_loss": [...], "val_cosine_sim": [...]}``
        and *best_metrics* contains the best validation loss and
        corresponding epoch.
    """
    teacher_model.eval()
    student_model.train()

    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss": [],
        "val_cosine_sim": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    global_step = 0
    val_metrics: Dict[str, float] = {"val_loss": 0.0, "cosine_similarity": 0.0, "mse": 0.0}

    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0.0
        num_batches: int = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
        )

        for batch_idx, texts in enumerate(pbar):
            global_step += 1
            num_batches += 1

            loss = _train_step(
                teacher_model=teacher_model,
                teacher_tokenizer=teacher_tokenizer,
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                texts=texts,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                max_grad_norm=max_grad_norm,
            )

            epoch_loss += loss

            if scheduler is not None:
                scheduler.step()

            if batch_idx % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss:.6f}"})

            if global_step % eval_interval == 0:
                val_metrics = evaluate(
                    teacher_model=teacher_model,
                    teacher_tokenizer=teacher_tokenizer,
                    student_model=student_model,
                    student_tokenizer=student_tokenizer,
                    val_loader=val_loader,
                    loss_fn=loss_fn,
                    device=device,
                )
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_cosine_sim"].append(val_metrics["cosine_similarity"])

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    best_epoch = epoch
                    _save_checkpoint(
                        student_model,
                        checkpoint_dir / "best_student_model.pt",
                        epoch=epoch,
                        val_loss=val_metrics["val_loss"],
                        is_best=True,
                    )

                student_model.train()

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_epoch_loss)

        logger.info(
            "Epoch %d/%d — train_loss=%.6f",
            epoch,
            num_epochs,
            avg_epoch_loss,
        )

        _save_checkpoint(
            student_model,
            checkpoint_dir / "last_student_model.pt",
            epoch=epoch,
            val_loss=val_metrics.get("val_loss", 0.0) if epoch > 1 else 0.0,
            is_best=False,
        )

    best_metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }

    return history, best_metrics


def _train_step(
    teacher_model: torch.nn.Module,
    teacher_tokenizer,
    student_model: torch.nn.Module,
    student_tokenizer,
    texts: List[str],
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """
    Single training step: forward through teacher (no grad), forward
    through student (with grad), compute loss, backprop, clip, step.
    """
    # --- Teacher forward (no grad) ---
    with torch.no_grad():
        teacher_emb = _encode_batch(
            teacher_model, teacher_tokenizer, texts, device
        )

    # --- Student forward (with grad) ---
    student_emb = _encode_batch(
        student_model, student_tokenizer, texts, device
    )

    # --- Loss ---
    loss = loss_fn(teacher_emb, student_emb)

    # --- Backprop ---
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        student_model.parameters(), max_grad_norm
    )
    optimizer.step()

    return loss.item()


def _encode_batch(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Tokenize texts and return L2-normalised mean-pooled embeddings."""
    encoded = tokenizer(
        texts,
        max_length=NLPConfig.MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded)
    pooled = mean_pooling(outputs, encoded["attention_mask"])
    return F.normalize(pooled, p=2, dim=1)


# ===================================================================
# Evaluation
# ===================================================================


@torch.no_grad()
def evaluate(
    teacher_model: torch.nn.Module,
    teacher_tokenizer,
    student_model: torch.nn.Module,
    student_tokenizer,
    val_loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the student on a validation set.

    Returns a dict with ``val_loss``, ``cosine_similarity``, and
    ``mse``.
    """
    teacher_model.eval()
    student_model.eval()

    total_loss: float = 0.0
    all_teacher_embs: List[torch.Tensor] = []
    all_student_embs: List[torch.Tensor] = []
    num_batches: int = 0

    for texts in val_loader:
        teacher_emb = _encode_batch(
            teacher_model, teacher_tokenizer, texts, device
        )
        student_emb = _encode_batch(
            student_model, student_tokenizer, texts, device
        )

        loss = loss_fn(teacher_emb, student_emb)
        total_loss += loss.item()
        num_batches += 1

        all_teacher_embs.append(teacher_emb.cpu())
        all_student_embs.append(student_emb.cpu())

    avg_loss = total_loss / max(num_batches, 1)

    teacher_concat = torch.cat(all_teacher_embs, dim=0)
    student_concat = torch.cat(all_student_embs, dim=0)

    cosine_sim = compute_cosine_similarity(teacher_concat, student_concat)
    mse_val = compute_mse(teacher_concat, student_concat)

    return {
        "val_loss": avg_loss,
        "cosine_similarity": cosine_sim,
        "mse": mse_val,
    }


# ===================================================================
# Checkpointing
# ===================================================================


def _save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    epoch: int,
    val_loss: float,
    is_best: bool,
) -> None:
    """Save student model checkpoint to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = "best" if is_best else "last"

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "teacher_model_name": DistillationConfig.TEACHER_MODEL_NAME,
                "student_model_name": DistillationConfig.STUDENT_MODEL_NAME,
                "loss_type": DistillationConfig.LOSS_TYPE,
                "embedding_dim": DistillationConfig.EMBEDDING_DIM,
            },
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )
    logger.info("Saved %s checkpoint -> %s (epoch=%d, val_loss=%.6f)", tag, path, epoch, val_loss)


def load_student_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> int:
    """
    Load a student checkpoint into *model* and return the epoch.

    Parameters
    ----------
    path : Path
        Path to ``.pt`` checkpoint file.
    model : torch.nn.Module
        Student model instance (must match architecture).
    device : torch.device

    Returns
    -------
    int
        The epoch at which the checkpoint was saved.
    """
    if not path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", 0)
    logger.info("Loaded student checkpoint from %s (epoch=%d)", path, epoch)
    return epoch


# ===================================================================
# Plotting
# ===================================================================


def plot_training_curves(
    history: Dict[str, List],
    save_path: Path,
) -> None:
    """
    Plot training loss, validation loss, and validation cosine similarity.

    Parameters
    ----------
    history : dict
        With keys ``train_loss``, ``val_loss``, ``val_cosine_sim``.
    save_path : Path
        Output path (e.g. ``models/nlp_student/training_curves.png``).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: loss curves
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    if history["val_loss"]:
        val_epochs = range(1, len(history["val_loss"]) + 1)
        ax.plot(val_epochs, history["val_loss"], marker="s", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Distillation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: cosine similarity
    ax = axes[1]
    if history["val_cosine_sim"]:
        val_epochs = range(1, len(history["val_cosine_sim"]) + 1)
        ax.plot(val_epochs, history["val_cosine_sim"], marker="s", color="green", label="Val Cosine Sim")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Teacher-Student Embedding Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved -> %s", save_path)


def save_comparison_table(
    comparison: dict,
    save_path: Path,
) -> None:
    """
    Save a teacher-vs-student comparison table as a CSV file.

    Parameters
    ----------
    comparison : dict
        Output from ``compare_models()``.
    save_path : Path
        Output CSV path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for side in ("Teacher", "Student"):
        if side in comparison:
            row = {"model": side}
            row.update(comparison[side])
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logger.info("Comparison table saved -> %s", save_path)


# ===================================================================
# Main entry point
# ===================================================================


def main(argv: List[str] | None = None) -> None:
    """
    Main entry point.

    1. Parse arguments.
    2. Resolve device.
    3. Load teacher (frozen) and student (trainable).
    4. Load cleaned reports.
    5. Create DataLoaders.
    6. Run distillation training loop.
    7. Save checkpoints, history, curves, and comparison table.
    """
    args = parse_args(argv)

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # --- Checkpoint directory ---
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if (checkpoint_dir / "best_student_model.pt").exists():
        logger.warning(
            "Checkpoint directory %s already contains best_student_model.pt — "
            "existing checkpoint will NOT be overwritten. "
            "Move or rename it to train a new student.",
            checkpoint_dir,
        )

    # --- Load teacher (frozen) ---
    logger.info("Loading teacher: %s", args.teacher_model_name)
    teacher_tokenizer, teacher_model, _ = build_encoder(
        model_name=args.teacher_model_name,
    )
    teacher_model.to(device)
    teacher_model.eval()

    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info("Teacher parameters: %s", f"{teacher_params:,}")

    # --- Load student (trainable) ---
    logger.info("Loading student: %s", args.student_model_name)
    student_tokenizer, student_model, _ = build_student_encoder(
        model_name=args.student_model_name,
        device=device,
        freeze=False,
    )
    student_model.to(device)

    student_params_total, student_params_trainable = count_parameters(student_model)
    logger.info(
        "Student parameters: %s total, %s trainable",
        f"{student_params_total:,}",
        f"{student_params_trainable:,}",
    )

    # --- Load dataset ---
    logger.info("Loading cleaned reports...")
    try:
        train_df = load_clean_reports("train")
        val_df = load_clean_reports("validation")
    except FileNotFoundError as exc:
        logger.error(
            "Dataset files not found. Run the NLP preprocessing notebooks "
            "first, or check that the files exist at %s. Error: %s",
            NLPConfig.PROCESSED_REPORTS_DIR,
            exc,
        )
        sys.exit(1)

    train_texts = train_df["clean_report"].dropna().tolist()
    val_texts = val_df["clean_report"].dropna().tolist()

    if len(train_texts) == 0 or len(val_texts) == 0:
        logger.error("No cleaned reports found — cannot train.")
        sys.exit(1)

    logger.info(
        "Loaded %d training reports and %d validation reports",
        len(train_texts),
        len(val_texts),
    )

    train_dataset = TextDataset(train_texts)
    val_dataset = TextDataset(val_texts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_texts,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_texts,
    )

    # --- Loss function ---
    loss_fn = get_loss_fn(args.loss_type)
    logger.info("Using loss function: %s", args.loss_type)

    # --- Optimiser & scheduler ---
    optimizer = AdamW(
        student_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Train ---
    logger.info("Starting distillation training...")
    start_time = time.time()

    history, best_metrics = train_student(
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        num_epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        max_grad_norm=args.max_grad_norm,
    )

    elapsed = time.time() - start_time
    logger.info(
        "Training completed in %.1f s. Best val loss: %.6f (epoch %d)",
        elapsed,
        best_metrics["best_val_loss"],
        best_metrics["best_epoch"],
    )

    # --- Save training history ---
    history_path = checkpoint_dir / "training_history.csv"
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    logger.info("Training history saved -> %s", history_path)

    # --- Plot curves ---
    curves_path = checkpoint_dir / "training_curves.png"
    plot_training_curves(history, curves_path)

    # --- Final evaluation on validation set ---
    logger.info("Running final evaluation...")
    val_metrics = evaluate(
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )
    logger.info(
        "Final validation — loss=%.6f, cosine_similarity=%.4f, mse=%.6f",
        val_metrics["val_loss"],
        val_metrics["cosine_similarity"],
        val_metrics["mse"],
    )

    # --- Comparison table ---
    logger.info("Generating comparison table...")
    student_model.eval()
    teacher_model.eval()

    comparison = compare_models(
        teacher_model=teacher_model,
        student_model=student_model,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        texts=val_texts[:100],
        device=device,
        num_latency_runs=30,
    )

    comparison_path = checkpoint_dir / "teacher_student_comparison.csv"
    save_comparison_table(comparison, comparison_path)

    logger.info(
        "Parameter reduction: %.1f%% | Size reduction: %.1f%%",
        comparison["Reduction"]["param_reduction_pct"],
        comparison["Reduction"]["size_reduction_pct"],
    )

    if "Speedup" in comparison:
        logger.info(
            "Inference speedup: %.2f×",
            comparison["Speedup"]["mean_speedup_x"],
        )

    if "Similarity" in comparison:
        logger.info(
            "Embedding similarity — cos=%.4f, mse=%.6f",
            comparison["Similarity"]["cosine_similarity"],
            comparison["Similarity"]["mse"],
        )

    logger.info("Distillation complete. All artifacts saved to %s", checkpoint_dir)


if __name__ == "__main__":
    main()
