"""Training and testing utilities for the spotter."""

from .testing import TestArtifact, evaluate_patchcore_experiment
from .training import TrainingArtifact, train_patchcore_experiment

__all__ = [
    "TestArtifact",
    "TrainingArtifact",
    "evaluate_patchcore_experiment",
    "train_patchcore_experiment",
]
from .spotter_losses import SpotterReconstructionLoss
from .spotter_training import train_spotter_model

__all__ = ["SpotterReconstructionLoss", "train_spotter_model"]
