"""
Configuration for Knowledge Distillation of the NLP encoder.

Defines teacher / student model choices, training hyper-parameters,
and an inference-time flag that allows the student to replace the
full-size teacher without touching any existing call site.

This config is completely independent of ``src/nlp_module/config.py``
and can be imported without side effects.

Author:
Nour Hossam
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DistillationConfig:
    """
    Frozen dataclass used as a read-only namespace for all distillation
    and student-inference settings.

    Every field has a sensible default so that the common case
    ``DistillationConfig.PARAM_NAME`` works without any user-side
    initialisation.
    """

    # ==========================================================
    # Model selection
    # ==========================================================

    TEACHER_MODEL_NAME: str = "dmis-lab/biobert-base-cased-v1.1"

    STUDENT_MODEL_NAME: str = "distilbert-base-cased"

    # ==========================================================
    # Training
    # ==========================================================

    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 10
    OPTIMIZER: str = "AdamW"
    SCHEDULER: str = "linear"
    WARMUP_STEPS: int = 0
    WEIGHT_DECAY: float = 0.01

    # ==========================================================
    # Loss
    # ==========================================================

    LOSS_TYPE: str = "mse"
    LOSS_TYPE_CHOICES: tuple = ("mse", "cosine", "mse_cosine")

    # ==========================================================
    # Device / hardware
    # ==========================================================

    DEVICE: str = "auto"

    # ==========================================================
    # Checkpointing & logging
    # ==========================================================

    CHECKPOINT_DIR: Path = Path("models/nlp_student")
    LOG_INTERVAL: int = 10
    EVAL_INTERVAL: int = 50
    MAX_GRAD_NORM: float = 1.0

    # ==========================================================
    # Embedding output (must match teacher)
    # ==========================================================

    EMBEDDING_DIM: int = 768

    # ==========================================================
    # Inference-time flag
    #
    # When USE_STUDENT_MODEL is True the lightweight student encoder
    # is used instead of the teacher for all text-feature extraction.
    # Existing callers do NOT need to change -- they keep calling
    # extract_text_features() and the flag is checked internally.
    # ==========================================================

    USE_STUDENT_MODEL: bool = False
