from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.spotter import evaluate_patchcore_experiment, load_spotter_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PatchCore spotter checkpoint on the prepared test set.")
    parser.add_argument("--exp_name", required=True, help="Experiment name used for dataset and output folders.")
    parser.add_argument(
        "--config",
        default="src/config/spotter_patchcore.yaml",
        help="Path to YAML config with dataset/model defaults.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. By default experiments/spotter/<exp_name>/weights/patchcore.ckpt is used.",
    )
    parser.add_argument(
        "--prepare_if_missing",
        action="store_true",
        help="Prepare the dataset from raw_data if the prepared dataset is missing.",
    )
    parser.add_argument(
        "--force_prepare",
        action="store_true",
        help="Recreate the prepared dataset before evaluation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device for the predictor pass. Examples: auto, cpu, gpu.",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Optional override for anomalib engine accelerator during metric evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_spotter_config(args.config, workspace_root=WORKSPACE_ROOT)
    if args.accelerator is not None:
        config.engine.accelerator = args.accelerator

    artifact = evaluate_patchcore_experiment(
        config=config,
        exp_name=args.exp_name,
        checkpoint_path=args.checkpoint,
        prepare_if_missing=args.prepare_if_missing,
        force_prepare=args.force_prepare,
        device=args.device,
    )
    print(json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
