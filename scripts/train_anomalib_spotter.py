from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.spotter import load_anomalib_spotter_config, train_anomalib_spotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchCore-based anomaly spotter with anomalib.")
    parser.add_argument("--exp_name", required=True, help="Experiment name used for dataset and output folders.")
    parser.add_argument(
        "--config",
        default="src/config/anomalib_patchcore_spotter.yaml",
        help="Path to YAML config with dataset/model/training defaults.",
    )
    parser.add_argument(
        "--force_prepare",
        action="store_true",
        help="Recreate the prepared dataset before training.",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Override anomalib/lightning accelerator. Examples: auto, cpu, gpu.",
    )
    parser.add_argument(
        "--export_type",
        choices=("none", "torch", "onnx", "openvino"),
        default=None,
        help="Optional export format after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_anomalib_spotter_config(args.config, workspace_root=WORKSPACE_ROOT)

    if args.accelerator is not None:
        config.engine.accelerator = args.accelerator
    if args.export_type is not None:
        config.engine.export_type = args.export_type

    artifact = train_anomalib_spotter(
        config,
        args.exp_name,
        prepare_if_missing=True,
        force_prepare=args.force_prepare,
    )
    print(json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
