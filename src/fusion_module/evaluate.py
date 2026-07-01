"""
Model Evaluation & Explainability for CortexAI Fusion Module.

Implements NB05 as a runnable script and importable module.

Sections
--------
1. evaluate_all_splits()      — Accuracy, F1, Classification Report
2. plot_confusion_matrices()  — 3 splits side by side
3. plot_training_history()    — Loss + Accuracy curves
4. plot_calibration()         — Reliability diagrams per risk class
5. run_shap()                 — GradientExplainer on DecisionHead
6. plot_shap_importance()     — Global clinical feature importance bar chart
7. plot_shap_beeswarm()       — Beeswarm per risk class
8. plot_waterfall()           — Per-patient waterfall
9. analyze_representations()  — PCA + t-SNE of 256-d fusion embeddings
10. find_similar_patients()   — Cosine similarity retrieval
11. plot_confidence_dist()    — Confidence distribution: correct vs incorrect
12. run_evaluation()          — Full pipeline in one call

Usage
-----
    python -m fusion_module.evaluate

    # or with custom paths:
    python -m fusion_module.evaluate \
        --text-encoder biobert \
        --model-dir    /kaggle/input/.../models/fusion \
        --repr-dir     /kaggle/input/.../reports/fusion/representations \
        --fusion-dir   /kaggle/input/.../datasets/processed/fusion \
        --clinical-dir /kaggle/input/.../datasets/processed/clinical_features \
        --figure-dir   reports/figures/fusion \
        --result-dir   reports/results/fusion

Author: Ammar Kamal
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import FusionConfig
from .dataset import FusionDataset, load_fusion_split
from .fusion_model import ClinicalDecisionModel, build_model
from .train import assign_labels, compute_thresholds, get_clinical_columns

__all__ = [
    "EvaluationResult",
    "load_model_for_eval",
    "load_eval_data",
    "evaluate_all_splits",
    "plot_confusion_matrices",
    "plot_training_history",
    "plot_calibration",
    "run_shap",
    "plot_shap_importance",
    "plot_shap_beeswarm",
    "plot_waterfall",
    "analyze_representations",
    "find_similar_patients",
    "plot_confidence_dist",
    "run_evaluation",
]

SPLITS      = ("train", "validation", "test")
RISK_NAMES  = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLORS = ["#2a78d6", "#eda100", "#e63946"]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_for_eval(
    model_dir: Path,
    device:    torch.device,
) -> tuple[ClinicalDecisionModel, list[str], int]:
    """
    Load best_decision_model.pth and return (model, clinical_columns, clinical_dim).
    CLINICAL_COLUMNS always read from the checkpoint — never hardcoded.
    """
    ckpt = torch.load(
        model_dir / "best_decision_model.pth",
        map_location=device,
        weights_only=False,
    )
    clinical_columns: list[str] = ckpt["clinical_cols"]
    clinical_dim:     int       = ckpt["clinical_dim"]

    model = build_model(clinical_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Checkpoint epoch  : {ckpt['epoch']}")
    print(f"Best val accuracy : {ckpt['val_acc']:.4f}")
    print(f"Clinical dim      : {clinical_dim}")
    print(f"Clinical columns  : {clinical_columns}")

    return model, clinical_columns, clinical_dim


# ── Data loading ──────────────────────────────────────────────────────────────

def load_eval_data(
    fusion_dir:      Path,
    clinical_dir:    Path,
    model_dir:       Path,
    clinical_columns: list[str],
    batch_size:      int = 32,
) -> tuple[dict, dict]:
    """
    Load fusion features + clinical tables, assign labels, scale, build loaders.

    Returns (splits_data, loaders) where:
        splits_data[split] = {image, text, clinical DataFrame}
        loaders[split]     = DataLoader
    """
    scaler     = joblib.load(model_dir / "clinical_scaler.pkl")
    thresholds = json.loads((model_dir / "severity_thresholds.json").read_text())

    splits_data: dict = {}
    for split in SPLITS:
        fusion   = load_fusion_split(fusion_dir, split)
        clinical = pd.read_csv(clinical_dir / f"{split}_clinical_features.csv")
        clinical["severity_score"] = _compute_score(clinical, thresholds)
        clinical["risk_label"]     = clinical["severity_score"].apply(_risk_label)
        clinical[clinical_columns] = scaler.transform(clinical[clinical_columns])
        splits_data[split] = {
            "image":    fusion["image_features"],
            "text":     fusion["text_features"],
            "clinical": clinical,
        }

    for split, data in splits_data.items():
        vc = data["clinical"]["risk_label"].value_counts().sort_index()
        print(f"  {split:12}  Low={vc.get(0,0)}  Medium={vc.get(1,0)}  High={vc.get(2,0)}")

    loaders = {
        split: DataLoader(
            FusionDataset(
                splits_data[split]["image"],
                splits_data[split]["text"],
                splits_data[split]["clinical"][clinical_columns].values,
                splits_data[split]["clinical"]["risk_label"].values,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        for split in SPLITS
    }

    return splits_data, loaders


def _compute_score(df: pd.DataFrame, thresholds: dict) -> list[int]:
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


def _risk_label(score: int) -> int:
    if score <= 1: return 0
    if score <= 3: return 1
    return 2


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvaluationResult:
    """Container for labels, predictions, and probabilities per split."""
    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    def add(self, split: str, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> None:
        self.data[split] = {"labels": labels, "preds": preds, "probs": probs}

    def __getitem__(self, split: str) -> dict:
        return self.data[split]

    def items(self):
        return self.data.items()


def evaluate_all_splits(
    model:   ClinicalDecisionModel,
    loaders: dict,
    device:  torch.device,
) -> EvaluationResult:
    """Run inference on all splits and return EvaluationResult."""
    result = EvaluationResult()

    for split, loader in loaders.items():
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                logits, _ = model(
                    batch["image"].to(device),
                    batch["text"].to(device),
                    batch["clinical"].to(device),
                )
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(probs.argmax(axis=1).tolist())
                all_labels.extend(batch["label"].tolist())

        labels = np.array(all_labels)
        preds  = np.array(all_preds)
        probs  = np.array(all_probs)
        result.add(split, labels, preds, probs)

        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="weighted")
        print(f"{'='*55}")
        print(f"{split.upper():12}  Accuracy: {acc:.4f}  Weighted-F1: {f1:.4f}")
        print(classification_report(labels, preds, target_names=RISK_NAMES, digits=4))

    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    eval_result: EvaluationResult,
    figure_dir:  Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (split_name, res) in zip(axes, eval_result.items()):
        cm   = confusion_matrix(res["labels"], res["preds"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "Medium", "High"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        acc = accuracy_score(res["labels"], res["preds"])
        f1  = f1_score(res["labels"], res["preds"], average="weighted")
        ax.set_title(f"{split_name.capitalize()}\nAcc={acc:.4f}  F1={f1:.4f}", fontsize=12)
        ax.grid(False)
    plt.suptitle("Confusion Matrices — Clinical Decision Model", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → confusion_matrices.png")


def plot_training_history(
    model_dir:  Path,
    figure_dir: Path,
) -> None:
    history_df = pd.read_csv(model_dir / "training_history.csv")
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train", color="#2a78d6", lw=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"],   label="Val",   color="#e63946", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train", color="#2a78d6", lw=2)
    axes[1].plot(history_df["epoch"], history_df["val_acc"],   label="Val",   color="#e63946", lw=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Training History — ClinicalDecisionModel", fontsize=13)
    plt.tight_layout()
    plt.savefig(figure_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → training_history.png")


def plot_calibration(
    eval_result: EvaluationResult,
    figure_dir:  Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, cls_idx, cls_name in zip(axes, range(3), RISK_NAMES):
        for split_name, color in [("train", "#2a78d6"), ("validation", "#eda100"), ("test", "#e63946")]:
            res           = eval_result[split_name]
            binary_labels = (res["labels"] == cls_idx).astype(int)
            prob_cls      = res["probs"][:, cls_idx]
            if binary_labels.sum() == 0:
                continue
            fraction_pos, mean_pred = calibration_curve(
                binary_labels, prob_cls, n_bins=8, strategy="quantile"
            )
            ax.plot(mean_pred, fraction_pos, "o-", color=color, label=split_name, lw=1.5, ms=5)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration — {cls_name}", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Reliability Diagrams (per risk class)", fontsize=13)
    plt.tight_layout()
    plt.savefig(figure_dir / "calibration.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → calibration.png")


def plot_confidence_dist(
    eval_result: EvaluationResult,
    figure_dir:  Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (split_name, res) in zip(axes, eval_result.items()):
        confidence = res["probs"].max(axis=1)
        correct    = (res["labels"] == res["preds"])
        ax.hist(confidence[correct],  bins=20, alpha=0.7, color="#2a78d6", label="Correct",   edgecolor="white")
        ax.hist(confidence[~correct], bins=20, alpha=0.7, color="#e63946", label="Incorrect", edgecolor="white")
        ax.axvline(confidence.mean(), color="black", lw=1.5, linestyle="--",
                   label=f"Mean conf: {confidence.mean():.3f}")
        ax.set_xlabel("Max Softmax Probability")
        ax.set_ylabel("Patients")
        ax.set_title(f"{split_name.capitalize()} — Confidence Distribution", fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Confidence Distribution — Correct vs Incorrect Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(figure_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → confidence_distribution.png")


# ── SHAP ──────────────────────────────────────────────────────────────────────

class _DecisionHeadWrapper(nn.Module):
    """Wrap DecisionHead to accept a single concatenated tensor (B, 256+N)."""
    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x[:, :256], x[:, 256:])


def _collect_decision_inputs(
    model:   ClinicalDecisionModel,
    loader:  DataLoader,
    device:  torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect (unified_repr, clinical, labels) tensors from a loader."""
    all_repr, all_clin, all_lbl = [], [], []
    with torch.no_grad():
        for batch in loader:
            _, unified_repr = model(
                batch["image"].to(device),
                batch["text"].to(device),
                batch["clinical"].to(device),
            )
            all_repr.append(unified_repr.cpu())
            all_clin.append(batch["clinical"])
            all_lbl.append(batch["label"])
    return torch.cat(all_repr), torch.cat(all_clin), torch.cat(all_lbl)


def run_shap(
    model:            ClinicalDecisionModel,
    train_loader:     DataLoader,
    test_loader:      DataLoader,
    device:           torch.device,
    clinical_dim:     int,
    bg_samples:       int = 200,
) -> np.ndarray:
    """
    Run GradientExplainer on the DecisionHead.

    Returns
    -------
    shap_values : np.ndarray  shape (N_test, 256 + clinical_dim, 3)
        Axis 0 = test patients
        Axis 1 = features  (first 256 are repr dims, rest are clinical)
        Axis 2 = risk classes
    """
    train_repr, train_clin, _ = _collect_decision_inputs(model, train_loader, device)
    test_repr,  test_clin,  _ = _collect_decision_inputs(model, test_loader,  device)

    train_dh = torch.cat([train_repr, train_clin], dim=1)
    test_dh  = torch.cat([test_repr,  test_clin],  dim=1)

    print(f"DecisionHead input: train {train_dh.shape}  test {test_dh.shape}")
    print(f"  (256 repr + {clinical_dim} clinical = {256 + clinical_dim} total)")

    wrapper = _DecisionHeadWrapper(model.decision_head).to(device)
    wrapper.eval()

    # Sanity check wrapper output shape
    with torch.no_grad():
        test_out = wrapper(train_dh[:4].to(device))
    assert test_out.shape == (4, 3), f"Wrapper output shape {test_out.shape} != (4, 3)"

    torch.manual_seed(42)
    bg_idx     = torch.randperm(train_dh.shape[0])[:bg_samples]
    background = train_dh[bg_idx].to(device)

    explainer   = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(test_dh.to(device))
    # shap_values: list of 3 arrays (N, F)  →  stack to (N, F, 3)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=2)

    print(f"SHAP values shape: {shap_values.shape}  (N, F, classes)")
    return shap_values, test_clin


def plot_shap_importance(
    shap_values:      np.ndarray,
    clinical_columns: list[str],
    clinical_dim:     int,
    figure_dir:       Path,
    result_dir:       Path,
) -> None:
    """Global clinical feature importance bar chart."""
    # mean |SHAP| across patients (axis 0) and classes (axis 2)
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))   # (F,)
    clinical_shap = mean_abs_shap[-clinical_dim:]            # (clinical_dim,)

    assert len(clinical_columns) == len(clinical_shap), (
        f"CLINICAL_COLUMNS ({len(clinical_columns)}) != SHAP clinical dims ({len(clinical_shap)})"
    )

    shap_df = pd.DataFrame({
        "feature":    clinical_columns,
        "importance": clinical_shap,
    }).sort_values("importance", ascending=False)

    shap_df.to_csv(result_dir / "shap_clinical_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(shap_df["feature"][::-1], shap_df["importance"][::-1],
            color="#2a78d6", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Clinical Feature Importance (SHAP)\nAveraged across all risk classes & test patients", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_clinical_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → shap_clinical_importance.png")


def plot_shap_beeswarm(
    shap_values:      np.ndarray,
    test_clin:        torch.Tensor,
    clinical_columns: list[str],
    figure_dir:       Path,
) -> None:
    """Beeswarm scatter plot per risk class — clinical features only."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    np.random.seed(42)

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, RISK_NAMES)):
        sv        = shap_values[:, 256:, cls_idx]     # (N, clinical_dim)
        order     = np.argsort(np.abs(sv).mean(axis=0))[::-1][:12]
        sv_top    = sv[:, order]
        feat_top  = [clinical_columns[i] for i in order]
        feat_vals = test_clin.numpy()[:, order]

        for j in range(sv_top.shape[1]):
            y_jitter = np.random.normal(j, 0.08, size=sv_top.shape[0])
            ax.scatter(sv_top[:, j], y_jitter,
                       c=feat_vals[:, j], cmap="coolwarm",
                       s=15, alpha=0.7, linewidths=0)

        ax.set_yticks(range(len(feat_top)))
        ax.set_yticklabels(feat_top[::-1], fontsize=8)
        ax.axvline(0, color="black", lw=0.8, linestyle="--")
        ax.set_xlabel("SHAP value", fontsize=10)
        ax.set_title(cls_name, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("SHAP Beeswarm — Clinical Features per Risk Class\n(color = feature value: blue=low, red=high)", fontsize=12)
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → shap_beeswarm.png")


def plot_waterfall(
    patient_idx:      int,
    shap_values:      np.ndarray,
    eval_result:      EvaluationResult,
    clinical_columns: list[str],
    figure_dir:       Path,
    split:            str = "test",
    ax=None,
) -> None:
    """Waterfall plot for one patient — top clinical features only."""
    res      = eval_result[split]
    true_cls = int(res["labels"][patient_idx])
    pred_cls = int(res["preds"][patient_idx])
    conf     = float(res["probs"][patient_idx][pred_cls])

    sv_patient = shap_values[patient_idx, 256:, pred_cls]   # (clinical_dim,)
    order      = np.argsort(np.abs(sv_patient))[::-1][:10]
    feat_top   = [clinical_columns[i] for i in order]
    shap_top   = sv_patient[order]
    colors     = ["#e63946" if v > 0 else "#2a78d6" for v in shap_top]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(range(len(shap_top)), shap_top[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(feat_top)))
    ax.set_yticklabels(feat_top[::-1], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (impact on predicted class)", fontsize=10)
    ax.set_title(
        f"Patient #{patient_idx} — True: {RISK_NAMES[true_cls]}  |  Pred: {RISK_NAMES[pred_cls]}\n"
        f"Confidence: {conf:.3f}",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        plt.tight_layout()
        path = figure_dir / f"waterfall_patient_{patient_idx}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved → {path.name}")


def plot_waterfall_examples(
    shap_values:      np.ndarray,
    eval_result:      EvaluationResult,
    clinical_columns: list[str],
    figure_dir:       Path,
) -> None:
    """One waterfall per risk class (Low / Medium / High) side by side."""
    test_labels = eval_result["test"]["labels"]
    shown: dict[int, int] = {}
    for idx in range(len(test_labels)):
        cls = int(test_labels[idx])
        if cls not in shown:
            shown[cls] = idx
        if len(shown) == 3:
            break

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, (cls, idx) in zip(axes, sorted(shown.items())):
        plot_waterfall(idx, shap_values, eval_result, clinical_columns, figure_dir, ax=ax)

    plt.suptitle("SHAP Waterfall — One Patient per Risk Class (Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_waterfall_examples.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → shap_waterfall_examples.png")


# ── Representation analysis ───────────────────────────────────────────────────

def analyze_representations(
    repr_dir:   Path,
    figure_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA]:
    """
    Load saved unified representations, run PCA + t-SNE, plot results.

    Returns (all_repr_scaled, pca_coords, tsne_coords, scaler, pca_model)
    """
    repr_data: dict = {}
    for split in SPLITS:
        repr_data[split] = {
            "repr": np.load(repr_dir / f"{split}_unified_repr.npy"),
            "meta": pd.read_csv(repr_dir / f"{split}_repr_metadata.csv"),
        }
        print(f"{split:12}  repr shape: {repr_data[split]['repr'].shape}")

    all_repr   = np.concatenate([repr_data[s]["repr"] for s in SPLITS])
    all_labels = np.concatenate([repr_data[s]["meta"]["risk_label"].values for s in SPLITS])
    all_splits = np.concatenate([
        np.full(len(repr_data[s]["repr"]), i) for i, s in enumerate(SPLITS)
    ])

    scaler_repr = StandardScaler()
    all_repr_sc = scaler_repr.fit_transform(all_repr)

    pca_model = PCA(n_components=2, random_state=42)
    repr_pca  = pca_model.fit_transform(all_repr_sc)

    tsne_model = TSNE(n_components=2, perplexity=15, random_state=42, max_iter=1000)
    repr_tsne  = tsne_model.fit_transform(all_repr_sc)

    print(f"PCA explained variance (2 PCs): {pca_model.explained_variance_ratio_.sum():.3f}")

    # Plot: PCA + t-SNE colored by risk label
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, coords, title in zip(axes, [repr_pca, repr_tsne], ["PCA", "t-SNE"]):
        for cls_idx, (cls_name, color) in enumerate(zip(RISK_NAMES, RISK_COLORS)):
            mask = all_labels == cls_idx
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, s=40, alpha=0.75, edgecolors="white", linewidths=0.3,
                       label=f"{cls_name} (n={mask.sum()})")
        ax.set_title(f"{title} — Unified Fusion Representations\nColored by Risk Label", fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if title == "PCA":
            ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})")
        else:
            ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(figure_dir / "repr_pca_tsne_risk.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → repr_pca_tsne_risk.png")

    # Plot: PCA colored by split
    fig, ax = plt.subplots(figsize=(9, 7))
    for split_idx, (sname, scolor) in enumerate(zip(["Train", "Validation", "Test"], RISK_COLORS)):
        mask = all_splits == split_idx
        ax.scatter(repr_pca[mask, 0], repr_pca[mask, 1],
                   c=scolor, s=40, alpha=0.75, edgecolors="white",
                   linewidths=0.3, label=f"{sname} (n={mask.sum()})")
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — Unified Representations\nColored by Split", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(figure_dir / "repr_pca_splits.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → repr_pca_splits.png")

    return all_repr_sc, repr_pca, repr_tsne, scaler_repr, pca_model, repr_data


# ── Similar patient retrieval ─────────────────────────────────────────────────

def find_similar_patients(
    query_patient_id: str,
    repr_matrix:      np.ndarray,
    patient_ids:      list[str],
    top_k:            int = 5,
) -> pd.DataFrame:
    """Cosine similarity retrieval in the 256-d representation space."""
    if query_patient_id not in patient_ids:
        raise ValueError(f"{query_patient_id!r} not found in patient_ids.")
    q_idx = patient_ids.index(query_patient_id)
    sims  = cosine_similarity(repr_matrix[q_idx].reshape(1, -1), repr_matrix)[0]
    sims[q_idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_k]
    return pd.DataFrame({
        "rank":              range(1, top_k + 1),
        "patient_id":        [patient_ids[i] for i in top_idx],
        "cosine_similarity": [round(float(sims[i]), 4) for i in top_idx],
    })


def plot_retrieval(
    query_id:    str,
    repr_data:   dict,
    scaler_repr: StandardScaler,
    result_dir:  Path,
    figure_dir:  Path,
) -> None:
    """Run retrieval demo on validation split and plot PCA visualization."""
    val_repr_arr = repr_data["validation"]["repr"]
    val_meta     = repr_data["validation"]["meta"]
    val_pids     = val_meta["patient_id"].tolist()

    similar = find_similar_patients(query_id, val_repr_arr, val_pids)
    print(f"Query patient : {query_id}")
    print(similar.to_string(index=False))
    similar.to_csv(result_dir / "similar_patients_demo.csv", index=False)

    val_pca_sc = scaler_repr.transform(val_repr_arr)
    val_pca    = PCA(n_components=2, random_state=42).fit_transform(val_pca_sc)
    val_risk   = val_meta["risk_label"].values
    q_idx      = val_pids.index(query_id)
    top5_idx   = [val_pids.index(pid) for pid in similar["patient_id"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls_idx, (cls_name, color) in enumerate(zip(RISK_NAMES, RISK_COLORS)):
        mask = val_risk == cls_idx
        ax.scatter(val_pca[mask, 0], val_pca[mask, 1],
                   c=color, s=40, alpha=0.5, edgecolors="white", linewidths=0.3, label=cls_name)
    ax.scatter(val_pca[top5_idx, 0], val_pca[top5_idx, 1],
               c="#1baf7a", s=120, alpha=0.9, edgecolors="white", linewidths=0.8,
               label="Top-5 similar", zorder=4)
    for rank, idx in enumerate(top5_idx, 1):
        ax.annotate(f"#{rank}", (val_pca[idx, 0], val_pca[idx, 1]),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=8, ha="center", color="#1baf7a")
    ax.scatter(val_pca[q_idx, 0], val_pca[q_idx, 1],
               c="#e63946", s=200, zorder=5, marker="*",
               edgecolors="white", linewidths=1, label=f"Query: {query_id}")
    ax.set_title(f"Similar Patient Retrieval — PCA (Validation)\nQuery: {query_id}", fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(figure_dir / "similar_patient_retrieval.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → similar_patient_retrieval.png")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_evaluation(
    text_encoder:  str  = "biobert",
    fusion_dir:    Path | None = None,
    clinical_dir:  Path | None = None,
    model_dir:     Path | None = None,
    repr_dir:      Path | None = None,
    figure_dir:    Path | None = None,
    result_dir:    Path | None = None,
    batch_size:    int  = 32,
    shap_bg:       int  = 200,
) -> None:
    """Full NB05 pipeline in one call."""
    cfg = FusionConfig(TEXT_ENCODER=text_encoder)

    fusion_dir   = fusion_dir   or cfg.fusion_dir
    clinical_dir = clinical_dir or cfg.CLINICAL_DIR
    model_dir    = model_dir    or cfg.MODEL_DIR
    repr_dir     = repr_dir     or cfg.REPR_DIR
    figure_dir   = figure_dir   or Path("reports/figures/fusion")
    result_dir   = result_dir   or Path("reports/results/fusion")

    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # 1. Load model
    print("\n── Loading model ──")
    model, clinical_columns, clinical_dim = load_model_for_eval(model_dir, device)

    # 2. Load data
    print("\n── Loading data ──")
    splits_data, loaders = load_eval_data(
        fusion_dir, clinical_dir, model_dir, clinical_columns, batch_size
    )

    # 3. Evaluate
    print("\n── Evaluating ──")
    eval_result = evaluate_all_splits(model, loaders, device)

    # 4. Plots
    plot_confusion_matrices(eval_result, figure_dir)
    plot_training_history(model_dir, figure_dir)
    plot_calibration(eval_result, figure_dir)
    plot_confidence_dist(eval_result, figure_dir)

    # 5. SHAP
    print("\n── Running SHAP ──")
    shap_values, test_clin = run_shap(
        model, loaders["train"], loaders["test"], device, clinical_dim, shap_bg
    )
    plot_shap_importance(shap_values, clinical_columns, clinical_dim, figure_dir, result_dir)
    plot_shap_beeswarm(shap_values, test_clin, clinical_columns, figure_dir)
    plot_waterfall_examples(shap_values, eval_result, clinical_columns, figure_dir)

    # 6. Representation analysis
    print("\n── Representation analysis ──")
    _, _, _, scaler_repr, _, repr_data = analyze_representations(repr_dir, figure_dir)

    # 7. Similar patient retrieval
    val_pids = repr_data["validation"]["meta"]["patient_id"].tolist()
    plot_retrieval(val_pids[0], repr_data, scaler_repr, result_dir, figure_dir)

    # 8. Summary
    print("\n" + "=" * 65)
    print("Evaluation & Explainability Complete")
    print("=" * 65)
    for split_name, res in eval_result.items():
        acc = accuracy_score(res["labels"], res["preds"])
        f1  = f1_score(res["labels"], res["preds"], average="weighted")
        print(f"  {split_name:12}  Accuracy: {acc:.4f}  Weighted-F1: {f1:.4f}")
    print(f"\nFigures → {figure_dir}")
    print(f"Results → {result_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CortexAI fusion model.")
    parser.add_argument("--text-encoder",  default="biobert", choices=["biobert", "clinicalbert"])
    parser.add_argument("--fusion-dir",    type=Path, default=None)
    parser.add_argument("--clinical-dir",  type=Path, default=None)
    parser.add_argument("--model-dir",     type=Path, default=None)
    parser.add_argument("--repr-dir",      type=Path, default=None)
    parser.add_argument("--figure-dir",    type=Path, default=Path("reports/figures/fusion"))
    parser.add_argument("--result-dir",    type=Path, default=Path("reports/results/fusion"))
    parser.add_argument("--batch-size",    type=int,  default=32)
    parser.add_argument("--shap-bg",       type=int,  default=200)
    args = parser.parse_args()

    run_evaluation(
        text_encoder = args.text_encoder,
        fusion_dir   = args.fusion_dir,
        clinical_dir = args.clinical_dir,
        model_dir    = args.model_dir,
        repr_dir     = args.repr_dir,
        figure_dir   = args.figure_dir,
        result_dir   = args.result_dir,
        batch_size   = args.batch_size,
        shap_bg      = args.shap_bg,
    )


if __name__ == "__main__":
    main()
