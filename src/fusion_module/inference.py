"""
Inference API for the CortexAI Fusion Module.

Provides:
    FusionPredictor   — load checkpoint once, call predict() many times
    predict_case()    — convenience one-call function for the UI

Usage
-----
    from fusion_module.inference import predict_case

    result = predict_case(
        image_features = image_feats,   # np.ndarray (256,)
        text_features  = text_feats,    # np.ndarray (768,)
        clinical_row   = clinical_dict, # dict of clinical feature values
        patient_id     = "BraTS20_Training_001",
    )
    print(result.predicted_label)   # "High Risk"
    print(result.confidence)        # 0.87

Author: Ammar Kamal
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import json
import numpy as np
import pandas as pd
import torch

from .config import FusionConfig
from .fusion_model import ClinicalDecisionModel, build_model
from .train import assign_labels, get_clinical_columns

__all__ = [
    "FusionPrediction",
    "FusionPredictor",
    "load_predictor",
    "predict_case",
]

RISK_NAMES = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


@dataclass
class FusionPrediction:
    """Structured result of one fusion inference call."""

    predicted_class: int
    predicted_label: str
    confidence:      float
    probabilities:   dict[str, float]
    patient_id:      str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id":      self.patient_id,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "confidence":      round(self.confidence, 4),
            "probabilities":   {k: round(v, 4) for k, v in self.probabilities.items()},
        }


class FusionPredictor:
    """
    Load the trained ClinicalDecisionModel once and run inference.

    Parameters
    ----------
    checkpoint_path : Path | None
        Path to best_decision_model.pth.  Defaults to models/fusion/best_decision_model.pth.
    config : FusionConfig | None
    device : str | None   — "cuda", "cpu", or None (auto)
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        config:          FusionConfig | None = None,
        device:          str | None = None,
    ) -> None:
        self.config = config or FusionConfig()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt_path = checkpoint_path or (self.config.MODEL_DIR / "best_decision_model.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Fusion checkpoint not found: {ckpt_path}\n"
                "Run train.py first."
            )

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.clinical_columns: list[str] = ckpt["clinical_cols"]
        clinical_dim: int                = ckpt["clinical_dim"]

        self.model = build_model(clinical_dim).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # Load scaler
        scaler_path = self.config.MODEL_DIR / "clinical_scaler.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        # Load thresholds
        thr_path = self.config.MODEL_DIR / "severity_thresholds.json"
        self.thresholds = json.loads(thr_path.read_text()) if thr_path.exists() else None

    # ── Single patient ────────────────────────────────────────────────────────

    def predict(
        self,
        image_features: np.ndarray | torch.Tensor,
        text_features:  np.ndarray | torch.Tensor,
        clinical_row:   dict[str, float] | pd.Series | np.ndarray,
        patient_id:     str | None = None,
    ) -> FusionPrediction:
        """
        Predict clinical risk for one patient.

        Parameters
        ----------
        image_features : (256,)   — from extract_image_features()
        text_features  : (768,)   — from NLP pipeline
        clinical_row   : dict or array of clinical feature values
                         (must contain the same columns as CLINICAL_COLUMNS)
        patient_id     : optional identifier

        Returns
        -------
        FusionPrediction
        """
        img_t = _to_batch(image_features, self.device)   # (1, 256)
        txt_t = _to_batch(text_features,  self.device)   # (1, 768)
        clin_t = self._prepare_clinical(clinical_row)    # (1, N)

        with torch.no_grad():
            logits, _ = self.model(img_t, txt_t, clin_t)
            probs     = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_cls   = int(probs.argmax())
        confidence = float(probs[pred_cls])

        return FusionPrediction(
            predicted_class = pred_cls,
            predicted_label = RISK_NAMES[pred_cls],
            confidence      = confidence,
            probabilities   = {RISK_NAMES[i]: float(probs[i]) for i in range(3)},
            patient_id      = patient_id,
        )

    # ── Batch ─────────────────────────────────────────────────────────────────

    def predict_batch(
        self,
        image_features:  np.ndarray,
        text_features:   np.ndarray,
        clinical_matrix: np.ndarray,
        patient_ids:     list[str] | None = None,
    ) -> list[FusionPrediction]:
        """
        Predict for a batch of patients.

        Parameters
        ----------
        image_features  : (N, 256)
        text_features   : (N, 768)
        clinical_matrix : (N, clinical_dim) — already scaled
        patient_ids     : optional list of N IDs

        Returns
        -------
        list of FusionPrediction length N
        """
        img_t  = torch.as_tensor(image_features,  dtype=torch.float32).to(self.device)
        txt_t  = torch.as_tensor(text_features,   dtype=torch.float32).to(self.device)
        clin_t = torch.as_tensor(clinical_matrix, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(img_t, txt_t, clin_t)
            probs     = torch.softmax(logits, dim=1).cpu().numpy()

        results = []
        for i in range(len(probs)):
            pred_cls = int(probs[i].argmax())
            results.append(FusionPrediction(
                predicted_class = pred_cls,
                predicted_label = RISK_NAMES[pred_cls],
                confidence      = float(probs[i][pred_cls]),
                probabilities   = {RISK_NAMES[j]: float(probs[i][j]) for j in range(3)},
                patient_id      = patient_ids[i] if patient_ids else None,
            ))
        return results

    # ── Clinical preprocessing ────────────────────────────────────────────────

    def _prepare_clinical(
        self,
        clinical_row: dict[str, float] | pd.Series | np.ndarray,
    ) -> torch.Tensor:
        """
        Convert a clinical row dict / array to a scaled (1, N) tensor.

        If clinical_row is a dict, it is aligned to self.clinical_columns.
        If it is already a numpy array of the right shape, it is used as-is
        (caller is responsible for scaling).
        """
        if isinstance(clinical_row, np.ndarray):
            arr = clinical_row.astype(np.float32).reshape(1, -1)
        else:
            if isinstance(clinical_row, pd.Series):
                clinical_row = clinical_row.to_dict()
            arr = np.array(
                [float(clinical_row.get(c, 0.0)) for c in self.clinical_columns],
                dtype=np.float32,
            ).reshape(1, -1)

        if self.scaler is not None:
            arr = self.scaler.transform(arr).astype(np.float32)

        return torch.as_tensor(arr).to(self.device)


# ── Module-level helpers ──────────────────────────────────────────────────────

def load_predictor(
    checkpoint_path: Path | None = None,
    config:          FusionConfig | None = None,
) -> FusionPredictor:
    """Build and return a ready-to-use FusionPredictor."""
    return FusionPredictor(checkpoint_path=checkpoint_path, config=config)


def predict_case(
    image_features:  np.ndarray | torch.Tensor,
    text_features:   np.ndarray | torch.Tensor,
    clinical_row:    dict[str, float] | pd.Series | np.ndarray,
    patient_id:      str | None = None,
    checkpoint_path: Path | None = None,
    config:          FusionConfig | None = None,
) -> FusionPrediction:
    """
    One-call inference for Ibrahim's UI and Ahmed's XAI module.

    Parameters
    ----------
    image_features  : (256,)  from cv_module.predict.extract_image_features()
    text_features   : (768,)  from nlp_module pipeline
    clinical_row    : dict of clinical feature values (unscaled)
    patient_id      : optional patient identifier
    checkpoint_path : optional override
    config          : optional config override

    Returns
    -------
    FusionPrediction

    Example
    -------
    >>> result = predict_case(img_feats, txt_feats, clinical_dict, "BraTS20_Training_001")
    >>> print(result.predicted_label, result.confidence)
    """
    predictor = FusionPredictor(checkpoint_path=checkpoint_path, config=config)
    return predictor.predict(image_features, text_features, clinical_row, patient_id)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_batch(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.float()
    else:
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t.to(device)
