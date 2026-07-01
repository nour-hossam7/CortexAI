"""About — project architecture and team."""

from __future__ import annotations

import streamlit as st

from src.ui.bootstrap import setup_page
from src.ui.utils.disclaimer import render_disclaimer

setup_page("About", icon="ℹ️")

st.title("About CortexAI")
render_disclaimer()

st.markdown(
    """
    **CortexAI** is a multimodal clinical decision support system for brain tumor analysis,
    integrating MRI segmentation, radiology report NLP, and fusion-based risk stratification
    with explainable AI outputs.

    ### Architecture
    1. **Computer Vision** — MONAI SegResNet (4-channel BraTS preprocessing, sliding-window inference)
    2. **NLP** — BioBERT / ClinicalBERT mean-pooled report embeddings (768-d)
    3. **Fusion** — ClinicalDecisionModel (256-d unified representation + 13 clinical features → 3-class risk)
    4. **Explainability** — Grad-CAM (imaging), SHAP (clinical decision head), PCA / t-SNE (fusion space)

    ### Models
    | Component | Details |
    |-----------|---------|
    | SegResNet | 256-d bottleneck, ROI 128³, WT/TC/ET sub-regions |
    | BioBERT | `dmis-lab/biobert-base-cased-v1.1` |
    | ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` |
    | Fusion | Best validation accuracy 87.5% |

    ### Pipeline
    MRI → Segmentation → Feature Extraction → Clinical Report → ClinicalBERT → Fusion → Decision → Explainability → Report

    ### Authors
    | Name | Role |
    |------|------|
    | Nour Hossam | NLP |
    | Mariam Mohamed | Computer Vision |
    | Ammar Kamal | Fusion |
    | Ahmed Hossam | Explainable AI |
    | Ibrahim Mahmoud | UI & Integration |

    MIT License — Graduation Project, Artificial Intelligence.
    """
)
