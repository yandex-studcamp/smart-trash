"""Inference helpers for explicit spotter backends."""

from .anomalib_spotter import AnomalibSpotter, AnomalibSpotterPrediction
from .autoencoder_inference import calibrate_autoencoder_spotter, evaluate_autoencoder_spotter
from .autoencoder_spotter import AutoencoderSpotter, AutoencoderSpotterPrediction
from .base_spotter import BaseSpotter
from .visualization import prediction_category, save_prediction_visuals

__all__ = [
    "AnomalibSpotter",
    "AnomalibSpotterPrediction",
    "AutoencoderSpotter",
    "AutoencoderSpotterPrediction",
    "BaseSpotter",
    "calibrate_autoencoder_spotter",
    "evaluate_autoencoder_spotter",
    "prediction_category",
    "save_prediction_visuals",
]
