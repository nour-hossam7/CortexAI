"""Cached model loaders for Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st
import torch

from src.cv_module.predict import load_model as load_segresnet
from src.fusion_module.config import FusionConfig
from src.fusion_module.inference import FusionPredictor
from src.nlp_module.model import build_encoder
from src.ui.utils.paths import get_paths


@st.cache_resource(show_spinner="Loading SegResNet…")
def get_segresnet():
    paths = get_paths()
    ckpt = paths.segresnet_checkpoint
    if not ckpt.exists():
        raise FileNotFoundError(f"SegResNet checkpoint not found: {ckpt}")
    return load_segresnet(ckpt)


@st.cache_resource(show_spinner="Loading Fusion model…")
def get_fusion_predictor() -> FusionPredictor | None:
    paths = get_paths()
    if not paths.fusion_checkpoint.exists():
        return None
    try:
        return FusionPredictor(checkpoint_path=paths.fusion_checkpoint, config=FusionConfig())
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading BioBERT encoder…")
def get_biobert_encoder():
    return build_encoder(model_name="dmis-lab/biobert-base-cased-v1.1")


@st.cache_resource(show_spinner="Loading ClinicalBERT encoder…")
def get_clinicalbert_encoder():
    return build_encoder(model_name="emilyalsentzer/Bio_ClinicalBERT")


def get_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def load_thresholds() -> dict | None:
    path = get_paths().severity_thresholds
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_scaler():
    path = get_paths().clinical_scaler
    if path.exists():
        return joblib.load(path)
    return None
