from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spotter import load_spotter_config, save_spotter_config, train_spotter_model
from src.spotter.utils.spotter_utils import make_experiment_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the spotter DAAE model.")
    parser.add_argument(
        "--exp_name",
        required=True,
        help="Experiment name used under experiments/spotter/<exp_name>/",
    )
    parser.add_argument(
        "--config",
        default="src/config/spotter/train_spotter_daae.json",
        help="Path to the training config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_spotter_config(args.config)
    config.exp_name = args.exp_name

    experiment_paths = make_experiment_paths(args.exp_name)
    save_spotter_config(config, experiment_paths.config_path)

    summary = train_spotter_model(config=config, experiment_paths=experiment_paths)
    print(f"Spotter training finished for '{args.exp_name}'.")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best val loss: {summary['best_val_loss']:.6f}")
    print(f"Weights: {summary['best_weights_path']}")


if __name__ == "__main__":
    main()
