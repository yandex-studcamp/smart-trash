"""Spotter package with explicit autoencoder and anomalib APIs."""

from .config import (
    AnomalibSpotterConfig,
    AutoencoderSpotterConfig,
    load_anomalib_spotter_config,
    load_autoencoder_spotter_config,
    save_autoencoder_spotter_config,
)
from .data import (
    AnomalibPreparedDatasetArtifact,
    anomalib_dataset_is_prepared,
    prepare_anomalib_spotter_dataset,
)
from .inference import (
    AnomalibSpotter,
    AnomalibSpotterPrediction,
    AutoencoderSpotter,
    AutoencoderSpotterPrediction,
    BaseSpotter,
    calibrate_autoencoder_spotter,
    evaluate_autoencoder_spotter,
    prediction_category,
    save_prediction_visuals,
)
from .models import AutoencoderSpotterModel
from .train import (
    AnomalibEvaluationArtifact,
    AnomalibTrainingArtifact,
    AutoencoderReconstructionLoss,
    evaluate_anomalib_spotter,
    train_anomalib_spotter,
    train_autoencoder_spotter,
)

__all__ = [
    "AnomalibEvaluationArtifact",
    "AnomalibPreparedDatasetArtifact",
    "AnomalibSpotter",
    "AnomalibSpotterConfig",
    "AnomalibSpotterPrediction",
    "AnomalibTrainingArtifact",
    "AutoencoderReconstructionLoss",
    "AutoencoderSpotter",
    "AutoencoderSpotterConfig",
    "AutoencoderSpotterModel",
    "AutoencoderSpotterPrediction",
    "BaseSpotter",
    "anomalib_dataset_is_prepared",
    "calibrate_autoencoder_spotter",
    "evaluate_anomalib_spotter",
    "evaluate_autoencoder_spotter",
    "load_anomalib_spotter_config",
    "load_autoencoder_spotter_config",
    "prediction_category",
    "prepare_anomalib_spotter_dataset",
    "save_autoencoder_spotter_config",
    "save_prediction_visuals",
    "train_anomalib_spotter",
    "train_autoencoder_spotter",
]
