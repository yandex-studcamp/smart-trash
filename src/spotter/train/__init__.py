"""Training and evaluation utilities for explicit spotter backends."""

from .anomalib_evaluation import AnomalibEvaluationArtifact, evaluate_anomalib_spotter
from .anomalib_training import AnomalibTrainingArtifact, build_anomalib_engine, train_anomalib_spotter
from .autoencoder_losses import AutoencoderReconstructionLoss
from .autoencoder_training import train_autoencoder_spotter

__all__ = [
    "AnomalibEvaluationArtifact",
    "AnomalibTrainingArtifact",
    "AutoencoderReconstructionLoss",
    "build_anomalib_engine",
    "evaluate_anomalib_spotter",
    "train_anomalib_spotter",
    "train_autoencoder_spotter",
]
