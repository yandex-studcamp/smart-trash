from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.spotter import AnomalibSpotter, load_anomalib_spotter_config, save_prediction_visuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PatchCore spotter inference on a single image.")
    parser.add_argument("--exp_name", required=True, help="Experiment name used to resolve the checkpoint path.")
    parser.add_argument("--image_path", required=True, help="Path to the image that should be scored.")
    parser.add_argument(
        "--config",
        default="src/config/anomalib_patchcore_spotter.yaml",
        help="Path to YAML config with model and path defaults.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. By default experiments/spotter/<exp_name>/weights/patchcore.ckpt is used.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device. Examples: auto, cpu, gpu.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory for saved visualization artifacts. By default experiments/spotter/<exp_name>/inference/<image_stem> is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_anomalib_spotter_config(args.config, workspace_root=WORKSPACE_ROOT)

    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image was not found: {image_path}")

    checkpoint_path = Path(args.checkpoint or config.checkpoint_path_for(args.exp_name)).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")

    predictor = AnomalibSpotter(checkpoint_path=checkpoint_path, config=config, device=args.device)
    prediction = predictor.predict_details(image_path)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else config.run_root_for(args.exp_name) / "inference" / image_path.stem
    save_prediction_visuals(
        image_path=image_path,
        prediction=prediction,
        output_dir=output_dir,
        metadata={
            "exp_name": args.exp_name,
            "checkpoint_path": str(checkpoint_path),
        },
    )

    payload = {
        "exp_name": args.exp_name,
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "score": None if prediction.score is None else float(prediction.score),
        "score_threshold": None if prediction.score_threshold is None else float(prediction.score_threshold),
        "raw_score_threshold": None if prediction.raw_score_threshold is None else float(prediction.raw_score_threshold),
        "label": None if prediction.label is None else int(prediction.label),
        "is_anomaly": prediction.is_anomaly,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
