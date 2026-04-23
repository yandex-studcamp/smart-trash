from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.spotter import load_spotter_config, prepare_spotter_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare anomaly spotter dataset from local good/bad frames.")
    parser.add_argument("--exp_name", required=True, help="Experiment name. Dataset will be written under prepared_root/exp_name.")
    parser.add_argument(
        "--config",
        default="src/config/spotter_patchcore.yaml",
        help="Path to YAML config with dataset/model defaults.",
    )
    parser.add_argument("--force", action="store_true", help="Recreate the prepared dataset if it already exists.")
    parser.add_argument(
        "--copy_mode",
        choices=("copy", "hardlink"),
        default=None,
        help="Override file materialization strategy for the prepared dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_spotter_config(args.config, workspace_root=WORKSPACE_ROOT)
    if args.copy_mode is not None:
        config.dataset.copy_mode = args.copy_mode

    artifact = prepare_spotter_dataset(config, args.exp_name, force=args.force)
    print(json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
