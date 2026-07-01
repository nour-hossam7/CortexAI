"""
Inference utilities for TextBraTS report embedding extraction.

Provides:

- On-demand text feature extraction for a single report (new patient,
  or re-embedding with a different MODEL_NAME)
- The same extraction logic validated end-to-end in Notebooks 06a/06b,
  refactored into a reusable function instead of a notebook loop

Author:
Nour Hossam
"""

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .model import build_encoder, mean_pooling
from .preprocessing import clean_report

__all__ = [
    "extract_text_features",
    "extract_text_features_batch",
]


def extract_text_features(
    report_text: str,
    model_name: str | None = None,
    tokenizer=None,
    model=None,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Extract a fixed-size feature vector for a single radiology report.

    This is the text-module counterpart to cv_module.predict.
    extract_image_features() — both return a single per-patient feature
    vector ready for the fusion module. Pipeline:

        clean_report() → tokenize (max_length, padding, truncation) →
        frozen encoder forward pass → attention-mask mean pooling →
        L2 normalize

    Exactly matches Notebooks 06a/06b, so a report passed through here
    is directly comparable to the precomputed train/validation/test
    embeddings on disk (dataset.load_text_embeddings) — same model,
    same pooling, same normalization.

    Parameters
    ----------
    report_text : str
        Raw (uncleaned) radiology report text.
    model_name : str | None
        Defaults to Config.MODEL_NAME. Ignored if `tokenizer`/`model`
        are both provided.
    tokenizer, model, device : optional
        Pass these in (from model.build_encoder()) to avoid reloading
        the encoder from HuggingFace on every call — important when
        extracting features for many reports in a loop or a UI backend.
        If omitted, a fresh encoder is built for this call alone.

    Returns
    -------
    np.ndarray
        Shape (Config.EMBEDDING_DIM,) — L2-normalized mean-pooled
        embedding for this report.
    """
    if report_text is None:
        raise ValueError("report_text cannot be None.")

    report_text = str(report_text)

    if not report_text.strip():
        raise ValueError("report_text is empty.")
    
    if tokenizer is None or model is None:
        tokenizer, model, device = build_encoder(model_name=model_name)
    elif device is None:
        device = next(model.parameters()).device

    cleaned = clean_report(report_text)

    encoded = tokenizer(
        cleaned,
        max_length=Config.MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        pooled = mean_pooling(outputs, encoded["attention_mask"])

    pooled = pooled.squeeze()
    pooled = F.normalize(pooled, p=2, dim=0)

    return pooled.cpu().numpy().astype(np.float32)


def extract_text_features_batch(
    report_texts: list[str],
    model_name: str | None = None,
) -> np.ndarray:
    """
    Extract feature vectors for many reports, reusing a single loaded
    encoder instance.

    Convenience wrapper around extract_text_features() for batch jobs
    (e.g. re-embedding a whole new cohort with a different MODEL_NAME)
    — loads the tokenizer/model once instead of once per report.

    Note: this calls the encoder once per report (not a padded batch
    forward pass), matching the exact per-report loop validated in
    Notebooks 06a/06b. A true batched forward pass would be faster for
    large cohorts but was not what produced the saved embeddings —
    keep this version unless you also re-validate that batched and
    per-report outputs match bit-for-bit (padding interacts with
    attention in ways that can introduce tiny numerical differences
    across batch sizes).

    Parameters
    ----------
    report_texts : list[str]
        Raw (uncleaned) radiology report texts.
    model_name : str | None
        Defaults to Config.MODEL_NAME.

    Returns
    -------
    np.ndarray
        Shape (len(report_texts), Config.EMBEDDING_DIM).
    """

    if len(report_texts) == 0:
        return np.empty(
            (0, Config.EMBEDDING_DIM),
            dtype=np.float32,
        )

    tokenizer, model, device = build_encoder(
        model_name=model_name
    )

    features = [
        extract_text_features(
            text,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        for text in report_texts
    ]

    return np.asarray(features, dtype=np.float32)
