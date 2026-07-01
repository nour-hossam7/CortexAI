"""
CortexAI Fusion Model Architecture.

Classes
-------
ImageProjection      : (B, 256) → (B, 128)
TextProjection       : (B, 768) → (B, 128)
FusionBlock          : (B, 128), (B, 128) → (B, 256)   [concat → MLP]
FusionEncoder        : image + text → unified repr (B, 256)
DecisionHead         : repr (256) + clinical (N) → logits (B, 3)
ClinicalDecisionModel: full end-to-end model

Architecture (identical to NB02 / NB04 so checkpoints are compatible):

    Image  (256) ──► ImageProjection ──► (128) ──┐
                                                   ├─ cat (256) ──► FusionBlock ──► (256)
    Text   (768) ──► TextProjection  ──► (128) ──┘                                   │
                                                                                      ▼
                                              Clinical (N) ──────────────► DecisionHead ──► (3)

Author: Ammar Kamal
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "ImageProjection",
    "TextProjection",
    "FusionBlock",
    "FusionEncoder",
    "DecisionHead",
    "ClinicalDecisionModel",
    "build_model",
]


class ImageProjection(nn.Module):
    """Project SegResNet bottleneck features (256) → shared space (128)."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class TextProjection(nn.Module):
    """Project BioBERT / ClinicalBERT embeddings (768) → shared space (128)."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.LayerNorm(128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class FusionBlock(nn.Module):
    """Concatenate image + text projections (256) → unified repr (256)."""

    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(256, 256), nn.LayerNorm(256),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([image, text], dim=1))


class FusionEncoder(nn.Module):
    """
    Project image (256) + text (768) into a shared 256-d latent space.

    Parameters
    ----------
    None — architecture is fixed to match NB02 / NB04 checkpoints.

    Input
    -----
    image : (B, 256)
    text  : (B, 768)

    Output
    ------
    repr  : (B, 256)
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_projection = ImageProjection()
        self.text_projection  = TextProjection()
        self.fusion_block     = FusionBlock()

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.fusion_block(
            self.image_projection(image),
            self.text_projection(text),
        )


class DecisionHead(nn.Module):
    """
    MLP classifier: unified_repr (256) + clinical (N) → 3 risk logits.

    Parameters
    ----------
    fusion_dim   : int  — output dim of FusionEncoder (256)
    clinical_dim : int  — number of clinical features (varies after NB03 drops)
    hidden_dim   : int  — first hidden layer size (128)
    num_classes  : int  — 3 (Low / Medium / High risk)
    """

    def __init__(
        self,
        fusion_dim:   int = 256,
        clinical_dim: int = 13,
        hidden_dim:   int = 128,
        num_classes:  int = 3,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + clinical_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        fusion_repr: torch.Tensor,
        clinical:    torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(torch.cat([fusion_repr, clinical], dim=1))


class ClinicalDecisionModel(nn.Module):
    """
    Full end-to-end multimodal clinical decision support model.

    Forward pass returns BOTH logits and the unified representation so
    downstream tasks (SHAP, retrieval, clustering) can use the embedding
    without an extra forward pass.

    Parameters
    ----------
    clinical_dim : int — number of clinical features

    Input
    -----
    image    : (B, 256)
    text     : (B, 768)
    clinical : (B, clinical_dim)

    Output
    ------
    logits       : (B, 3)    — raw class scores
    unified_repr : (B, 256)  — fusion embedding for downstream use
    """

    def __init__(self, clinical_dim: int) -> None:
        super().__init__()
        self.fusion_encoder = FusionEncoder()
        self.decision_head  = DecisionHead(
            fusion_dim   = 256,
            clinical_dim = clinical_dim,
            hidden_dim   = 128,
            num_classes  = 3,
        )

    def forward(
        self,
        image:    torch.Tensor,
        text:     torch.Tensor,
        clinical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unified_repr = self.fusion_encoder(image, text)
        logits       = self.decision_head(unified_repr, clinical)
        return logits, unified_repr


def build_model(clinical_dim: int) -> ClinicalDecisionModel:
    """Build a ClinicalDecisionModel for the given clinical feature count."""
    return ClinicalDecisionModel(clinical_dim=clinical_dim)
