from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spotter import SpotterConfig, run_spotter_evaluation
from src.spotter.utils.spotter_utils import deep_update, load_json, make_experiment_paths


DEFAULT_TEST_CONFIG = PROJECT_ROOT / "src" / "config" / "spotter" / "test_spotter_daae.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the spotter DAAE model.")
    parser.add_argument(
        "--exp_name",
        required=True,
        help="Experiment name used under experiments/spotter/<exp_name>/",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config to merge on top of the experiment config.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional path to checkpoint weights. Defaults to best_spotter_daae.pt in the experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_paths = make_experiment_paths(args.exp_name)

    base_config_path = experiment_paths.config_path if experiment_paths.config_path.exists() else DEFAULT_TEST_CONFIG
    config_payload = load_json(base_config_path)
    if args.config:
        deep_update(config_payload, load_json(args.config))

    config = SpotterConfig.from_dict(config_payload)
    config.exp_name = args.exp_name

    summary = run_spotter_evaluation(
        config=config,
        experiment_paths=experiment_paths,
        weights_path=args.weights,
    )
    print(f"Spotter evaluation finished for '{args.exp_name}'.")
    print(f"F1: {summary['f1']:.4f}")
    print(f"Image threshold: {summary['image_threshold']:.6f}")
    print(f"Artifacts: {experiment_paths.artifacts_dir}")


if __name__ == "__main__":
    main()
