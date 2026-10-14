"""
Stage2 Release Module: Value-Aware Visual Token Release with Calibrated Risk Control

Core components:
- ReleaseEffect: Computes exact set-level deletion effect ε(R)
- CheckpointManager: Multi-checkpoint progressive release
- LTTCalibrator: Learn-then-Test risk calibration
"""

from .release_effect import ReleaseEffect, compute_release_effect
from .checkpoint_manager import (
    CheckpointManager,
    CheckpointConfig,
    create_checkpoint_manager,
)
from .calibration import LTTCalibrator, CalibrationConfig
from .utils import extract_question_rows, prepare_attention_mask

__version__ = "0.1.0"

__all__ = [
    "ReleaseEffect",
    "compute_release_effect",
    "CheckpointManager",
    "CheckpointConfig",
    "create_checkpoint_manager",
    "LTTCalibrator",
    "CalibrationConfig",
    "extract_question_rows",
    "prepare_attention_mask",
]
