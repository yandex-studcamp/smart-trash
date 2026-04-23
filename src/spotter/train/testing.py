from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from ..config import SpotterConfig
from ..data import dataset_is_prepared, prepare_spotter_dataset, build_folder_datamodule
from ..inference import SpotterPrediction, TorchSpotterPredictor
from ..models import build_patchcore_model
from .training import build_spotter_engine


@dataclass(slots=True)
class TestArtifact:
    exp_name: str
    checkpoint_path: Path
    metrics_path: Path
    predictions_path: Path
    anomalib_metrics: list[dict[str, Any]]
    custom_metrics: dict[str, Any]
    num_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_name": self.exp_name,
            "checkpoint_path": str(self.checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "predictions_path": str(self.predictions_path),
            "anomalib_metrics": self.anomalib_metrics,
            "custom_metrics": self.custom_metrics,
            "num_samples": self.num_samples,
        }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _prediction_to_row(path: Path, gt_label: int, split_name: str, prediction: SpotterPrediction) -> dict[str, Any]:
    return {
        "image_path": str(path),
        "split": split_name,
        "gt_label": gt_label,
        "pred_label": prediction.label,
        "is_anomaly": prediction.is_anomaly,
        "score": _safe_float(prediction.score),
        "score_threshold": _safe_float(prediction.score_threshold),
        "raw_score_threshold": _safe_float(prediction.raw_score_threshold),
    }


def _collect_prediction_rows(config: SpotterConfig, exp_name: str, predictor: TorchSpotterPredictor) -> list[dict[str, Any]]:
    dataset_root = config.dataset_root_for(exp_name)
    rows: list[dict[str, Any]] = []

    normal_dir = dataset_root / config.dataset.test_good_dir
    anomaly_dir = dataset_root / config.dataset.test_anomaly_dir

    for prediction in predictor.predict_directory(normal_dir):
        rows.append(_prediction_to_row(Path(prediction.image_path), 0, "good", prediction))
    for prediction in predictor.predict_directory(anomaly_dir):
        rows.append(_prediction_to_row(Path(prediction.image_path), 1, "anomaly", prediction))

    return rows


def _compute_custom_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_labels = [int(row["gt_label"]) for row in rows]
    pred_labels = [int(row["pred_label"]) if row["pred_label"] is not None else 0 for row in rows]
    scores = [float(row["score"]) if row["score"] is not None else 0.0 for row in rows]
    threshold = next((row["score_threshold"] for row in rows if row["score_threshold"] is not None), None)
    raw_threshold = next((row["raw_score_threshold"] for row in rows if row["raw_score_threshold"] is not None), None)

    metrics: dict[str, Any] = {
        "num_samples": len(rows),
        "num_normal": sum(label == 0 for label in gt_labels),
        "num_anomaly": sum(label == 1 for label in gt_labels),
        "accuracy": float(accuracy_score(gt_labels, pred_labels)),
        "precision": float(precision_score(gt_labels, pred_labels, zero_division=0)),
        "recall": float(recall_score(gt_labels, pred_labels, zero_division=0)),
        "f1": float(f1_score(gt_labels, pred_labels, zero_division=0)),
        "mean_score": float(mean(scores)) if scores else None,
        "score_threshold": threshold,
        "raw_score_threshold": raw_threshold,
    }

    if len(set(gt_labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(gt_labels, scores))
    else:
        metrics["roc_auc"] = None

    return metrics


def evaluate_patchcore_experiment(
    config: SpotterConfig,
    exp_name: str,
    *,
    checkpoint_path: str | Path | None = None,
    prepare_if_missing: bool = False,
    force_prepare: bool = False,
    device: str = "auto",
) -> TestArtifact:
    if prepare_if_missing and (force_prepare or not dataset_is_prepared(config, exp_name)):
        prepare_spotter_dataset(config, exp_name, force=force_prepare)

    dataset_root = config.dataset_root_for(exp_name)
    if not dataset_is_prepared(config, exp_name):
        raise FileNotFoundError(
            f"Prepared dataset for experiment '{exp_name}' was not found in {dataset_root}. "
            "Run the prepare script first or enable prepare_if_missing."
        )

    checkpoint_path = Path(checkpoint_path or config.checkpoint_path_for(exp_name)).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")

    evaluation_root = config.evaluation_root_for(exp_name)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    metrics_path = evaluation_root / "test_metrics.json"
    predictions_path = evaluation_root / "test_predictions.csv"

    predictor = TorchSpotterPredictor(checkpoint_path=checkpoint_path, config=config, device=device)
    rows = _collect_prediction_rows(config, exp_name, predictor)

    custom_metrics = _compute_custom_metrics(rows)

    datamodule = build_folder_datamodule(config, dataset_root, exp_name)
    engine = build_spotter_engine(config, config.run_root_for(exp_name))
    model = build_patchcore_model(config, evaluator=True, visualizer=False)
    anomalib_metrics = [dict(item) for item in engine.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)]

    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [
            "image_path",
            "split",
            "gt_label",
            "pred_label",
            "is_anomaly",
            "score",
            "score_threshold",
            "raw_score_threshold",
        ])
        writer.writeheader()
        writer.writerows(rows)

    artifact = TestArtifact(
        exp_name=exp_name,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        anomalib_metrics=anomalib_metrics,
        custom_metrics=custom_metrics,
        num_samples=len(rows),
    )
    metrics_path.write_text(json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact
