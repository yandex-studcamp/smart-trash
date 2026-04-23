from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .predictor import SpotterPrediction


def normalize_visual_map(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return None
    array = np.squeeze(array)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros(array.shape, dtype=np.uint8)
    finite_values = array[finite_mask]
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if max_value - min_value < 1e-12:
        normalized = np.zeros(array.shape, dtype=np.float32)
    else:
        normalized = (array - min_value) / (max_value - min_value)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def heatmap_rgb(normalized_map: np.ndarray) -> np.ndarray:
    values = normalized_map.astype(np.float32) / 255.0
    red = values
    green = np.clip(1.0 - np.abs(values - 0.5) * 2.0, 0.0, 1.0)
    blue = np.clip(1.0 - values, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1) * 255.0


def prediction_category(gt_label: int, pred_label: int) -> str:
    if gt_label == 1 and pred_label == 1:
        return "true_positive"
    if gt_label == 0 and pred_label == 0:
        return "true_negative"
    if gt_label == 0 and pred_label == 1:
        return "false_positive"
    return "false_negative"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def save_prediction_visuals(
    *,
    image_path: str | Path,
    prediction: SpotterPrediction,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_image = Image.open(image_path).convert("RGB")
    source_image.save(output_dir / "input.png")

    anomaly_map = normalize_visual_map(prediction.anomaly_map)
    if anomaly_map is not None:
        anomaly_image = Image.fromarray(anomaly_map, mode="L")
        resized_map = anomaly_image.resize(source_image.size, Image.Resampling.BILINEAR)
        resized_map.save(output_dir / "anomaly_map.png")

        heatmap = Image.fromarray(heatmap_rgb(np.asarray(resized_map, dtype=np.uint8)).astype(np.uint8), mode="RGB")
        heatmap.save(output_dir / "heatmap.png")

        overlay = Image.blend(source_image, heatmap, alpha=0.45)
        overlay.save(output_dir / "overlay.png")

    pred_mask = normalize_visual_map(prediction.pred_mask)
    if pred_mask is not None:
        mask_image = Image.fromarray(pred_mask, mode="L")
        mask_image = mask_image.resize(source_image.size, Image.Resampling.NEAREST)
        mask_image.save(output_dir / "pred_mask.png")

    payload = {
        "image_path": str(image_path),
        "score": _to_float(prediction.score),
        "score_threshold": _to_float(prediction.score_threshold),
        "raw_score_threshold": _to_float(prediction.raw_score_threshold),
        "label": None if prediction.label is None else int(prediction.label),
        "is_anomaly": prediction.is_anomaly,
    }
    if metadata:
        payload.update(metadata)
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir
