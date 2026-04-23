from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from ..config.spotter_config import SpotterConfig
from ..data.spotter_dataset import (
    SpotterEvaluationDataset,
    build_eval_records,
    load_spotter_image_tensor,
)
from ..models.spotter_model import SpotterDAAE
from ..utils.spotter_utils import ExperimentPaths, load_json, save_json, select_device


def tensor_to_rgb_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def compute_residual_map(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    residual = torch.abs(original - reconstructed)
    if residual.ndim == 4:
        return residual.mean(dim=1)
    return residual.mean(dim=0)


def reduce_anomaly_score(
    residual_map: torch.Tensor,
    reduction: str,
    quantile: float,
) -> float:
    flattened = residual_map.reshape(-1)
    if reduction == "mean":
        return float(flattened.mean().item())
    if reduction == "quantile":
        return float(torch.quantile(flattened, quantile).item())
    return float(flattened.max().item())


def extract_residual_boxes(
    residual_map: torch.Tensor,
    pixel_threshold: float,
    blur_kernel_size: int,
    morph_kernel_size: int,
    min_contour_area: int,
) -> tuple[list[dict[str, int | float]], np.ndarray]:
    residual_np = residual_map.detach().cpu().numpy().astype(np.float32)
    residual_uint8 = np.clip(residual_np * 255.0, 0, 255).astype(np.uint8)

    if blur_kernel_size > 1:
        blur_kernel_size = _ensure_odd(blur_kernel_size)
        residual_uint8 = cv2.GaussianBlur(residual_uint8, (blur_kernel_size, blur_kernel_size), 0)

    _, binary_mask = cv2.threshold(
        residual_uint8,
        int(round(pixel_threshold * 255.0)),
        255,
        cv2.THRESH_BINARY,
    )

    if morph_kernel_size > 1:
        morph_kernel_size = _ensure_odd(morph_kernel_size)
        kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[dict[str, int | float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_contour_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        boxes.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(width),
                "h": int(height),
                "area": area,
            }
        )
    boxes.sort(key=lambda box: float(box["area"]), reverse=True)
    return boxes, binary_mask


def calibrate_image_threshold(scores: list[float], labels: list[int]) -> dict[str, Any]:
    score_array = np.asarray(scores, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int32)
    if len(np.unique(label_array)) < 2:
        default_threshold = float(score_array.mean()) if len(score_array) else 0.0
        predictions = (score_array >= default_threshold).astype(np.int32)
        return {
            "best_threshold": default_threshold,
            "precision": float(precision_score(label_array, predictions, zero_division=0)),
            "recall": float(recall_score(label_array, predictions, zero_division=0)),
            "f1": float(f1_score(label_array, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(label_array, predictions)),
            "average_precision": 0.0,
            "curve_recall": [],
            "curve_precision": [],
        }

    curve_precision, curve_recall, thresholds = precision_recall_curve(label_array, score_array)
    if thresholds.size == 0:
        best_threshold = float(score_array.mean())
        selected_precision = float(curve_precision[0])
        selected_recall = float(curve_recall[0])
        selected_f1 = float(
            2 * selected_precision * selected_recall / max(selected_precision + selected_recall, 1e-8)
        )
    else:
        f1_values = 2 * curve_precision[:-1] * curve_recall[:-1] / np.clip(
            curve_precision[:-1] + curve_recall[:-1],
            1e-8,
            None,
        )
        best_index = int(np.nanargmax(f1_values))
        best_threshold = float(thresholds[best_index])
        selected_precision = float(curve_precision[best_index])
        selected_recall = float(curve_recall[best_index])
        selected_f1 = float(f1_values[best_index])

    predictions = (score_array >= best_threshold).astype(np.int32)
    return {
        "best_threshold": best_threshold,
        "precision": float(precision_score(label_array, predictions, zero_division=0)),
        "recall": float(recall_score(label_array, predictions, zero_division=0)),
        "f1": float(f1_score(label_array, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(label_array, predictions)),
        "average_precision": float(average_precision_score(label_array, score_array)),
        "curve_recall": curve_recall.tolist(),
        "curve_precision": curve_precision.tolist(),
        "selected_precision": selected_precision,
        "selected_recall": selected_recall,
        "selected_f1": selected_f1,
    }


def save_pr_curve(plot_path: Path, recall: list[float], precision: list[float], ap_score: float) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#0f766e", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Spotter PR Curve (AP={ap_score:.4f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def render_detection_artifact(
    original_tensor: torch.Tensor,
    reconstructed_tensor: torch.Tensor,
    residual_map: torch.Tensor,
    binary_mask: np.ndarray,
    boxes: list[dict[str, int | float]],
    output_path: Path,
    title: str,
) -> None:
    original_image = tensor_to_rgb_image(original_tensor)
    reconstructed_image = tensor_to_rgb_image(reconstructed_tensor)
    overlay_image = original_image.copy()

    for box in boxes:
        x = int(box["x"])
        y = int(box["y"])
        width = int(box["w"])
        height = int(box["h"])
        cv2.rectangle(overlay_image, (x, y), (x + width, y + height), (255, 64, 64), 2)

    plt.figure(figsize=(10, 8))
    plt.suptitle(title)

    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(residual_map.detach().cpu().numpy(), cmap="inferno")
    plt.title("Residual")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(overlay_image)
    plt.imshow(binary_mask, cmap="gray", alpha=0.25)
    plt.title("Detections")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_spotter_checkpoint(
    config: SpotterConfig,
    weights_path: Path,
    device: torch.device,
) -> SpotterDAAE:
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = SpotterDAAE(config.model).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_calibrated_threshold(experiment_paths: ExperimentPaths) -> float:
    calibration_path = experiment_paths.artifacts_dir / "calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file was not found: {calibration_path}. "
            "Run training with validation anomalies or calibrate threshold before test."
        )
    calibration_payload = load_json(calibration_path)
    if "best_threshold" not in calibration_payload:
        raise KeyError(f"'best_threshold' is missing in calibration file: {calibration_path}")
    return float(calibration_payload["best_threshold"])


def run_spotter_calibration(
    config: SpotterConfig,
    experiment_paths: ExperimentPaths,
    weights_path: str | Path | None = None,
) -> dict[str, Any]:
    if not config.data.val_normal_dir or not config.data.val_anomaly_dir:
        raise ValueError("Validation calibration requires both val_normal_dir and val_anomaly_dir.")

    device = select_device(config.device)
    resolved_weights_path = Path(weights_path) if weights_path else experiment_paths.weights_dir / "best_spotter_daae.pt"
    if not resolved_weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_weights_path}")

    model = load_spotter_checkpoint(config, resolved_weights_path, device)
    records = build_eval_records(
        normal_dir=config.data.val_normal_dir,
        anomaly_dir=config.data.val_anomaly_dir,
        image_extensions=config.data.image_extensions,
        max_eval_normal_images=config.data.max_eval_normal_images,
        max_eval_anomaly_images=config.data.max_eval_anomaly_images,
    )
    prediction_rows = _predict_records(
        config=config,
        model=model,
        device=device,
        records=records,
    )
    scores = [float(row["score"]) for row in prediction_rows]
    labels = [int(row["label"]) for row in prediction_rows]
    calibration = calibrate_image_threshold(scores=scores, labels=labels)
    threshold = float(calibration["best_threshold"])

    for row in prediction_rows:
        row["predicted_label"] = int(float(row["score"]) >= threshold)
        row["boxes_json"] = json.dumps(row["boxes"], ensure_ascii=False)

    predictions_path = experiment_paths.artifacts_dir / "calibration_predictions.csv"
    _save_predictions_csv(predictions_path, prediction_rows)
    if calibration["curve_precision"] and calibration["curve_recall"]:
        save_pr_curve(
            plot_path=experiment_paths.artifacts_dir / "calibration_pr_curve.png",
            recall=calibration["curve_recall"],
            precision=calibration["curve_precision"],
            ap_score=float(calibration["average_precision"]),
        )

    save_json(calibration, experiment_paths.artifacts_dir / "calibration.json")
    summary = {
        "weights_path": str(resolved_weights_path),
        "device": str(device),
        "samples": len(prediction_rows),
        "normal_samples": sum(1 for row in prediction_rows if int(row["label"]) == 0),
        "anomaly_samples": sum(1 for row in prediction_rows if int(row["label"]) == 1),
        "image_threshold": threshold,
        "precision": calibration["precision"],
        "recall": calibration["recall"],
        "f1": calibration["f1"],
        "accuracy": calibration["accuracy"],
        "average_precision": calibration["average_precision"],
        "predictions_path": str(predictions_path),
    }
    save_json(summary, experiment_paths.artifacts_dir / "calibration_summary.json")
    return summary


def run_spotter_evaluation(
    config: SpotterConfig,
    experiment_paths: ExperimentPaths,
    weights_path: str | Path | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    device = select_device(config.device)
    resolved_weights_path = Path(weights_path) if weights_path else experiment_paths.weights_dir / "best_spotter_daae.pt"
    if not resolved_weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_weights_path}")

    model = load_spotter_checkpoint(config, resolved_weights_path, device)
    records = build_eval_records(
        normal_dir=config.data.eval_normal_dir,
        anomaly_dir=config.data.eval_anomaly_dir,
        image_extensions=config.data.image_extensions,
        max_eval_normal_images=config.data.max_eval_normal_images,
        max_eval_anomaly_images=config.data.max_eval_anomaly_images,
    )
    prediction_rows = _predict_records(
        config=config,
        model=model,
        device=device,
        records=records,
    )
    applied_threshold = float(threshold) if threshold is not None else load_calibrated_threshold(experiment_paths)

    for row in prediction_rows:
        row["predicted_label"] = int(float(row["score"]) >= applied_threshold)
        row["boxes_json"] = json.dumps(row["boxes"], ensure_ascii=False)

    predictions_path = experiment_paths.artifacts_dir / "test_predictions.csv"
    _save_predictions_csv(predictions_path, prediction_rows)
    test_metrics = _compute_binary_metrics(prediction_rows)

    _save_visualizations(
        model=model,
        config=config,
        device=device,
        prediction_rows=prediction_rows,
        output_dir=experiment_paths.artifacts_dir / "examples",
        threshold=applied_threshold,
    )

    summary = {
        "weights_path": str(resolved_weights_path),
        "device": str(device),
        "samples": len(prediction_rows),
        "normal_samples": sum(1 for row in prediction_rows if int(row["label"]) == 0),
        "anomaly_samples": sum(1 for row in prediction_rows if int(row["label"]) == 1),
        "image_threshold": applied_threshold,
        "pixel_threshold": config.inference.pixel_threshold,
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "accuracy": test_metrics["accuracy"],
        "average_precision": test_metrics["average_precision"],
        "predictions_path": str(predictions_path),
    }
    save_json(summary, experiment_paths.artifacts_dir / "test_metrics.json")
    return summary


def _predict_records(
    config: SpotterConfig,
    model: SpotterDAAE,
    device: torch.device,
    records: list[tuple[Path, int]],
) -> list[dict[str, Any]]:
    dataset = SpotterEvaluationDataset(records, image_size=config.data.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
    )

    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"]
            paths = batch["path"]

            reconstructions = model(images)
            residual_maps = compute_residual_map(images, reconstructions)

            for index, image_path in enumerate(paths):
                residual_map = residual_maps[index]
                score = reduce_anomaly_score(
                    residual_map=residual_map,
                    reduction=config.inference.image_score_reduction,
                    quantile=config.inference.image_score_quantile,
                )
                boxes, _ = extract_residual_boxes(
                    residual_map=residual_map,
                    pixel_threshold=config.inference.pixel_threshold,
                    blur_kernel_size=config.inference.blur_kernel_size,
                    morph_kernel_size=config.inference.morph_kernel_size,
                    min_contour_area=config.inference.min_contour_area,
                )
                prediction_rows.append(
                    {
                        "path": image_path,
                        "label": int(labels[index]),
                        "score": score,
                        "box_count": len(boxes),
                        "largest_box_area": float(boxes[0]["area"]) if boxes else 0.0,
                        "boxes": boxes,
                    }
                )
    return prediction_rows


def _compute_binary_metrics(prediction_rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in prediction_rows], dtype=np.int32)
    predictions = np.asarray([int(row["predicted_label"]) for row in prediction_rows], dtype=np.int32)
    scores = np.asarray([float(row["score"]) for row in prediction_rows], dtype=np.float32)
    average_precision = float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "average_precision": average_precision,
    }


def _save_predictions_csv(output_path: Path, prediction_rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "label",
        "predicted_label",
        "score",
        "box_count",
        "largest_box_area",
        "boxes_json",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow({field: row[field] for field in fieldnames})


def _save_visualizations(
    model: SpotterDAAE,
    config: SpotterConfig,
    device: torch.device,
    prediction_rows: list[dict[str, Any]],
    output_dir: Path,
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_per_split = config.inference.max_visualizations_per_split

    for split_name, label in (("normal", 0), ("anomaly", 1), ("false_positive", "fp"), ("false_negative", "fn")):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

    normal_rows = sorted(
        (row for row in prediction_rows if int(row["label"]) == 0),
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:max_per_split]
    anomaly_rows = sorted(
        (row for row in prediction_rows if int(row["label"]) == 1),
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:max_per_split]
    false_positive_rows = sorted(
        (row for row in prediction_rows if int(row["label"]) == 0 and int(row["predicted_label"]) == 1),
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:max_per_split]
    false_negative_rows = sorted(
        (row for row in prediction_rows if int(row["label"]) == 1 and int(row["predicted_label"]) == 0),
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:max_per_split]

    selection_map = {
        "normal": normal_rows,
        "anomaly": anomaly_rows,
        "false_positive": false_positive_rows,
        "false_negative": false_negative_rows,
    }
    for split_name, rows in selection_map.items():
        for index, row in enumerate(rows):
            image_tensor = load_spotter_image_tensor(row["path"], config.data.image_size).unsqueeze(0).to(device)
            with torch.no_grad():
                reconstruction = model(image_tensor)[0].cpu()
            original = image_tensor[0].cpu()
            residual_map = compute_residual_map(original, reconstruction)
            boxes, binary_mask = extract_residual_boxes(
                residual_map=residual_map,
                pixel_threshold=config.inference.pixel_threshold,
                blur_kernel_size=config.inference.blur_kernel_size,
                morph_kernel_size=config.inference.morph_kernel_size,
                min_contour_area=config.inference.min_contour_area,
            )
            title = (
                f"label={row['label']} pred={row['predicted_label']} "
                f"score={float(row['score']):.4f} thr={threshold:.4f}"
            )
            filename = f"{index:02d}_{Path(row['path']).stem}.png"
            render_detection_artifact(
                original_tensor=original,
                reconstructed_tensor=reconstruction,
                residual_map=residual_map,
                binary_mask=binary_mask,
                boxes=boxes,
                output_path=output_dir / split_name / filename,
                title=title,
            )


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1
