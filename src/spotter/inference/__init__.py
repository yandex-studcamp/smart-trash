"""Inference helpers for the anomalib spotter."""

from .predictor import SpotterPrediction, TorchSpotterPredictor
from .visualization import prediction_category, save_prediction_visuals

__all__ = [
    "SpotterPrediction",
    "TorchSpotterPredictor",
    "prediction_category",
    "save_prediction_visuals",
]
