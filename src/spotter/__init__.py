"""Spotter package for anomaly detection."""

from .config.spotter_config import SpotterConfig, load_spotter_config, save_spotter_config
from .inference.spotter_inference import run_spotter_evaluation
from .inference.spotter_predictor import SpotterPredictor
from .models.spotter_model import SpotterDAAE
from .train.spotter_training import train_spotter_model

from .config import SpotterConfig, load_spotter_config
from .data import PreparedDatasetArtifact, dataset_is_prepared, prepare_spotter_dataset
from .inference import SpotterPrediction, TorchSpotterPredictor, prediction_category, save_prediction_visuals
from .train import TestArtifact, TrainingArtifact, evaluate_patchcore_experiment, train_patchcore_experiment

__all__ = [
    "PreparedDatasetArtifact",
    "SpotterConfig",
    "SpotterPrediction",
    "TestArtifact",
    "TorchSpotterPredictor",
    "TrainingArtifact",
    "dataset_is_prepared",
    "evaluate_patchcore_experiment",
    "SpotterPredictor",
    "SpotterDAAE",
    "load_spotter_config",
    "prediction_category",
    "prepare_spotter_dataset",
    "save_prediction_visuals",
    "train_patchcore_experiment",
    "run_spotter_evaluation",
    "save_spotter_config",
    "train_spotter_model",
]
