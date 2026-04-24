from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from ..config import AnomalibSpotterConfig
from ..data import (
    anomalib_dataset_is_prepared,
    build_anomalib_datamodule,
    prepare_anomalib_spotter_dataset,
)
from ..inference import AnomalibSpotter, AnomalibSpotterPrediction, prediction_category, save_prediction_visuals
from ..models import build_anomalib_patchcore_model
from .anomalib_training import build_anomalib_engine


@dataclass(slots=True)
class AnomalibEvaluationArtifact:
    exp_name: str
    checkpoint_path: Path
    metrics_path: Path
    predictions_path: Path
    examples_root: Path
    anomalib_metrics: list[dict[str, Any]]
    custom_metrics: dict[str, Any]
    num_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_name": self.exp_name,
            "checkpoint_path": str(self.checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "predictions_path": str(self.predictions_path),
            "examples_root": str(self.examples_root),
            "anomalib_metrics": self.anomalib_metrics,
            "custom_metrics": self.custom_metrics,
            "num_samples": self.num_samples,
        }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _prediction_to_row(
    path: Path,
    gt_label: int,
    split_name: str,
    prediction: AnomalibSpotterPrediction,
) -> dict[str, Any]:
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


def _collect_prediction_rows(
        config: AnomalibSpotterConfig,
        exp_name: str,
        predictor: AnomalibSpotter,
) -> tuple[list[dict[str, Any]], dict[str, AnomalibSpotterPrediction]]:
    dataset_root = config.dataset_root_for(exp_name)
    rows: list[dict[str, Any]] = []
    predictions_by_path: dict[str, AnomalibSpotterPrediction] = {}

    normal_dir = dataset_root / config.dataset.test_good_dir
    anomaly_dir = dataset_root / config.dataset.test_anomaly_dir

    for prediction in predictor.predict_directory_details(normal_dir):
        image_path = Path(prediction.image_path)
        rows.append(_prediction_to_row(image_path, 0, "good", prediction))
        predictions_by_path[str(image_path)] = prediction
    for prediction in predictor.predict_directory_details(anomaly_dir):
        image_path = Path(prediction.image_path)
        rows.append(_prediction_to_row(image_path, 1, "anomaly", prediction))
        predictions_by_path[str(image_path)] = prediction

    return rows, predictions_by_path


def _save_visual_examples(
        rows: list[dict[str, Any]],
        predictions_by_path: dict[str, AnomalibSpotterPrediction],
        examples_root: Path,
) -> None:
    examples_root.mkdir(parents=True, exist_ok=True)

    for row in rows:
        image_path = Path(str(row["image_path"])).resolve()
        prediction = predictions_by_path.get(str(image_path))
        if prediction is None or not image_path.exists():
            continue

        stem = image_path.stem
        split_name = str(row["split"])
        category = prediction_category(int(row["gt_label"]), int(row["pred_label"]))
        sample_root = examples_root / category / split_name / stem
        save_prediction_visuals(
            image_path=image_path,
            prediction=prediction,
            output_dir=sample_root,
            metadata={
                "split": split_name,
                "gt_label": int(row["gt_label"]),
                "pred_label": int(row["pred_label"]),
                "category": category,
                "score": _safe_float(row["score"]),
                "score_threshold": _safe_float(row["score_threshold"]),
                "raw_score_threshold": _safe_float(row["raw_score_threshold"]),
            },
        )


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


def evaluate_anomalib_spotter(
        config: AnomalibSpotterConfig,
        exp_name: str,
        *,
        checkpoint_path: str | Path | None = None,
        prepare_if_missing: bool = False,
        force_prepare: bool = False,
        device: str = "auto",
) -> AnomalibEvaluationArtifact:
    if prepare_if_missing and (force_prepare or not anomalib_dataset_is_prepared(config, exp_name)):
        prepare_anomalib_spotter_dataset(config, exp_name, force=force_prepare)

    dataset_root = config.dataset_root_for(exp_name)
    if not anomalib_dataset_is_prepared(config, exp_name):
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
    examples_root = evaluation_root / "examples"

    predictor = AnomalibSpotter(checkpoint_path=checkpoint_path, config=config, device=device)
    rows, predictions_by_path = _collect_prediction_rows(config, exp_name, predictor)

    custom_metrics = _compute_custom_metrics(rows)

    datamodule = build_anomalib_datamodule(config, dataset_root, exp_name)
    engine = build_anomalib_engine(config, config.run_root_for(exp_name))
    model = build_anomalib_patchcore_model(config, evaluator=True, visualizer=False)
    anomalib_metrics = [dict(item) for item in
                        engine.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)]

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

    _save_visual_examples(rows, predictions_by_path, examples_root)

    artifact = AnomalibEvaluationArtifact(
        exp_name=exp_name,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        examples_root=examples_root,
        anomalib_metrics=anomalib_metrics,
        custom_metrics=custom_metrics,
        num_samples=len(rows),
    )
    metrics_path.write_text(json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact
