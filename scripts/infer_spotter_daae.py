from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spotter.inference.spotter_predictor import SpotterPredictor
from src.spotter.utils.spotter_utils import ensure_dir, make_experiment_paths, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spotter DAAE inference on a single image.")
    parser.add_argument(
        "--exp_name",
        required=True,
        help="Experiment name from experiments/spotter/<exp_name>/",
    )
    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, for example cpu or cuda.",
    )
    return parser.parse_args()


def infer_label_from_path(image_path: Path) -> int | None:
    lowered_parts = {part.lower() for part in image_path.parts}
    if "anomaly" in lowered_parts:
        return 1
    if "normal" in lowered_parts or "good" in lowered_parts:
        return 0
    return None


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    max_value = float(image.max()) if image.size else 0.0
    if max_value <= 0:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = image / max_value
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def save_grayscale_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="L").save(path)


def build_heatmap(anomaly_map_uint8: np.ndarray) -> np.ndarray:
    heatmap_bgr = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def build_overlay(input_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    blended = cv2.addWeighted(
        input_image.astype(np.float32),
        1.0 - alpha,
        heatmap.astype(np.float32),
        alpha,
        0.0,
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image was not found: {image_path}")

    predictor = SpotterPredictor.from_experiment(
        exp_name=args.exp_name,
        device=args.device,
    )
    prediction = predictor.predict_with_details(image_path)

    experiment_paths = make_experiment_paths(args.exp_name)
    output_dir = ensure_dir(experiment_paths.root_dir / "inference" / image_path.stem)

    input_image = prediction["original_image"]
    reconstructed_image = prediction["reconstructed_image"]
    residual_map = prediction["residual_map"].detach().cpu().numpy().astype(np.float32)
    binary_mask = prediction["binary_mask"].astype(np.uint8)

    anomaly_map_uint8 = normalize_to_uint8(residual_map)
    heatmap_image = build_heatmap(anomaly_map_uint8)
    overlay_image = build_overlay(input_image, heatmap_image)
    pred_mask_uint8 = np.where(binary_mask > 0, 255, 0).astype(np.uint8)

    save_rgb_image(output_dir / "input.png", input_image)
    save_rgb_image(output_dir / "reconstructed.png", reconstructed_image)
    save_grayscale_image(output_dir / "anomaly_map.png", anomaly_map_uint8)
    save_rgb_image(output_dir / "heatmap.png", heatmap_image)
    save_rgb_image(output_dir / "overlay.png", overlay_image)
    save_grayscale_image(output_dir / "pred_mask.png", pred_mask_uint8)

    raw_score = float(prediction["score"])
    raw_threshold = float(prediction["threshold"])
    normalized_score = raw_score / raw_threshold if raw_threshold > 0 else raw_score

    metadata: dict[str, Any] = {
        "image_path": str(image_path),
        "score": raw_score,
        "score_threshold": raw_threshold,
        "raw_score": raw_score,
        "raw_score_threshold": raw_threshold,
        "normalized_score": normalized_score,
        "label": infer_label_from_path(image_path),
        "is_anomaly": bool(prediction["is_anomaly"]),
        "exp_name": args.exp_name,
        "checkpoint_path": str(predictor.weights_path.resolve()),
        "config_path": str(predictor.config_path.resolve()),
        "box_count": int(prediction["box_count"]),
        "boxes": prediction["boxes"],
        "output_dir": str(output_dir),
    }
    save_json(metadata, output_dir / "metadata.json")

    print(f"[spotter] inference complete for exp='{args.exp_name}'")
    print(f"[spotter] image: {image_path}")
    print(f"[spotter] score={raw_score:.6f} threshold={raw_threshold:.6f} is_anomaly={metadata['is_anomaly']}")
    print(f"[spotter] artifacts: {output_dir}")


if __name__ == "__main__":
    main()
