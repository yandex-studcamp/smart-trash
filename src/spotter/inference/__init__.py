"""Inference helpers for spotter."""

from .predictor import SpotterPrediction, TorchSpotterPredictor
from .visualization import prediction_category, save_prediction_visuals

__all__ = [
    "SpotterPrediction",
    "TorchSpotterPredictor",
    "prediction_category",
    "save_prediction_visuals",
]
from .spotter_inference import run_spotter_evaluation

__all__ = ["run_spotter_evaluation"]
