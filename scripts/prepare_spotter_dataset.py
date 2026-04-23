from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


@dataclass(frozen=True)
class SplitWeights:
    train: float
    val: float
    test: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare spotter datasets for either anomalib PatchCore experiments or "
            "raw train/val/test splits used by the autoencoder branch."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=("anomalib", "autoencoder"),
        default=None,
        help=(
            "Optional explicit mode. If omitted, the script infers the mode from the "
            "provided arguments."
        ),
    )

    parser.add_argument(
        "--exp_name",
        default=None,
        help="Anomalib mode: experiment name written under prepared_root/exp_name.",
    )
    parser.add_argument(
        "--config",
        default="src/config/spotter_patchcore.yaml",
        help="Anomalib mode: path to YAML config with dataset/model defaults.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Anomalib mode: recreate the prepared dataset if it already exists.",
    )

    parser.add_argument(
        "--input_path",
        default=None,
        help="Autoencoder mode: path to raw dataset with anomaly/ and normal/ directories.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Autoencoder mode: output dataset path, for example data/spotter/my_dataset.",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=None,
        help="Autoencoder mode: relative train split size. Supports ratios like 0.7 or weights like 70.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=None,
        help="Autoencoder mode: relative validation split size. Supports ratios like 0.15 or weights like 15.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="Autoencoder mode: relative test split size. Supports ratios like 0.15 or weights like 15.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--copy_mode",
        choices=("copy", "hardlink"),
        default=None,
        help=(
            "Override file materialization strategy. "
            "If omitted, anomalib mode uses config.dataset.copy_mode and autoencoder mode uses copy."
        ),
    )
    return parser.parse_args()


def resolve_mode(args: argparse.Namespace) -> str:
    if args.mode is not None:
        return args.mode

    has_anomalib_args = args.exp_name is not None
    has_autoencoder_args = any(
        value is not None
        for value in (args.input_path, args.output_path, args.train_size, args.val_size, args.test_size)
    )

    if has_anomalib_args and has_autoencoder_args:
        raise ValueError(
            "Ambiguous arguments: provide either anomalib arguments (--exp_name) "
            "or autoencoder split arguments (--input_path/--output_path/...); not both."
        )
    if has_anomalib_args:
        return "anomalib"
    if has_autoencoder_args:
        return "autoencoder"
    raise ValueError(
        "Could not infer mode. Pass --exp_name for anomalib mode or "
        "--input_path/--output_path/--train_size/--val_size/--test_size for autoencoder mode."
    )


def collect_image_files(directories: list[Path]) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files: list[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                files.append(path)
    return files


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_split_weights(train_size: float, val_size: float, test_size: float) -> SplitWeights:
    raw_values = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }
    if any(value < 0 for value in raw_values.values()):
        raise ValueError("Split sizes must be non-negative.")
    total = sum(raw_values.values())
    if total <= 0:
        raise ValueError("At least one split size must be positive.")
    normalized = {key: value / total for key, value in raw_values.items()}
    if normalized["train"] <= 0:
        raise ValueError("Train split must be positive for normal images.")
    if normalized["val"] + normalized["test"] <= 0:
        raise ValueError("At least one of val/test must be positive.")
    return SplitWeights(**normalized)


def split_paths(paths: list[Path], first_ratio: float, second_ratio: float) -> tuple[list[Path], list[Path]]:
    if not paths:
        return [], []
    boundary = int(round(len(paths) * first_ratio / max(first_ratio + second_ratio, 1e-12)))
    boundary = max(0, min(boundary, len(paths)))
    return paths[:boundary], paths[boundary:]


def split_normal_paths(paths: list[Path], weights: SplitWeights) -> dict[str, list[Path]]:
    total = len(paths)
    train_end = int(round(total * weights.train))
    val_end = int(round(total * (weights.train + weights.val)))
    train_end = max(0, min(train_end, total))
    val_end = max(train_end, min(val_end, total))
    return {
        "train": paths[:train_end],
        "val": paths[train_end:val_end],
        "test": paths[val_end:],
    }


def materialize_files(paths: list[Path], destination_dir: Path, copy_mode: str) -> list[str]:
    ensure_dir(destination_dir)
    materialized_names: list[str] = []
    for source_path in paths:
        destination_path = destination_dir / source_path.name
        if destination_path.exists():
            destination_path.unlink()
        if copy_mode == "hardlink":
            try:
                destination_path.hardlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)
        materialized_names.append(source_path.name)
    return materialized_names


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_autoencoder_dataset(args: argparse.Namespace) -> None:
    required = {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Autoencoder mode requires: " + ", ".join(f"--{name}" for name in missing)
        )

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    normal_dir = input_path / "normal"
    anomaly_dir = input_path / "anomaly"

    if not normal_dir.exists():
        raise FileNotFoundError(f"Missing normal directory: {normal_dir}")
    if not anomaly_dir.exists():
        raise FileNotFoundError(f"Missing anomaly directory: {anomaly_dir}")

    weights = normalize_split_weights(args.train_size, args.val_size, args.test_size)
    rng = random.Random(args.seed)
    copy_mode = args.copy_mode or "copy"

    normal_paths = collect_image_files([normal_dir])
    anomaly_paths = collect_image_files([anomaly_dir])
    if not normal_paths:
        raise ValueError(f"No normal images found in {normal_dir}")
    if not anomaly_paths:
        raise ValueError(f"No anomaly images found in {anomaly_dir}")

    rng.shuffle(normal_paths)
    rng.shuffle(anomaly_paths)

    normal_splits = split_normal_paths(normal_paths, weights)
    anomaly_val_ratio = weights.val / (weights.val + weights.test) if (weights.val + weights.test) > 0 else 0.0
    anomaly_val_paths, anomaly_test_paths = split_paths(
        anomaly_paths,
        first_ratio=anomaly_val_ratio,
        second_ratio=1.0 - anomaly_val_ratio,
    )

    train_good_dir = output_path / "train" / "good"
    val_good_dir = output_path / "val" / "good"
    val_anomaly_dir = output_path / "val" / "anomaly"
    test_good_dir = output_path / "test" / "good"
    test_anomaly_dir = output_path / "test" / "anomaly"

    for target_dir in (
        train_good_dir,
        val_good_dir,
        val_anomaly_dir,
        test_good_dir,
        test_anomaly_dir,
    ):
        reset_directory(target_dir)

    train_good_files = materialize_files(normal_splits["train"], train_good_dir, copy_mode)
    val_good_files = materialize_files(normal_splits["val"], val_good_dir, copy_mode)
    val_anomaly_files = materialize_files(anomaly_val_paths, val_anomaly_dir, copy_mode)
    test_good_files = materialize_files(normal_splits["test"], test_good_dir, copy_mode)
    test_anomaly_files = materialize_files(anomaly_test_paths, test_anomaly_dir, copy_mode)

    manifest = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "seed": args.seed,
        "copy_mode": copy_mode,
        "split_weights": asdict(weights),
        "notes": {
            "normal_distribution": "normal images are split into train/val/test using the provided weights",
            "anomaly_distribution": "anomaly images are split only into val/test because anomaly samples are not used for training",
        },
        "counts": {
            "normal_total": len(normal_paths),
            "anomaly_total": len(anomaly_paths),
            "train_good": len(train_good_files),
            "val_good": len(val_good_files),
            "val_anomaly": len(val_anomaly_files),
            "test_good": len(test_good_files),
            "test_anomaly": len(test_anomaly_files),
        },
        "paths": {
            "train_good_dir": str(train_good_dir),
            "val_good_dir": str(val_good_dir),
            "val_anomaly_dir": str(val_anomaly_dir),
            "test_good_dir": str(test_good_dir),
            "test_anomaly_dir": str(test_anomaly_dir),
        },
        "files": {
            "train_good": train_good_files,
            "val_good": val_good_files,
            "val_anomaly": val_anomaly_files,
            "test_good": test_good_files,
            "test_anomaly": test_anomaly_files,
        },
    }
    save_json(manifest, output_path / "manifest.json")

    print(f"[spotter] dataset prepared at {output_path}")
    print(
        f"[spotter] train/good={len(train_good_files)} "
        f"val/good={len(val_good_files)} "
        f"val/anomaly={len(val_anomaly_files)} "
        f"test/good={len(test_good_files)} "
        f"test/anomaly={len(test_anomaly_files)}"
    )


def prepare_anomalib_dataset(args: argparse.Namespace) -> None:
    if args.exp_name is None:
        raise ValueError("Anomalib mode requires --exp_name.")

    from src.spotter import load_spotter_config, prepare_spotter_dataset

    config = load_spotter_config(args.config, workspace_root=WORKSPACE_ROOT)
    if args.copy_mode is not None:
        config.dataset.copy_mode = args.copy_mode

    artifact = prepare_spotter_dataset(config, args.exp_name, force=args.force)
    print(json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    mode = resolve_mode(args)
    if mode == "anomalib":
        prepare_anomalib_dataset(args)
        return
    prepare_autoencoder_dataset(args)
import sys
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
    parser.add_argument("--force", action="store_true", help="Recreate the prepared dataset if it already exists.")
def normalize_split_weights(train_size: float, val_size: float, test_size: float) -> SplitWeights:
    raw_values = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }
    if any(value < 0 for value in raw_values.values()):
        raise ValueError("Split sizes must be non-negative.")
    total = sum(raw_values.values())
    if total <= 0:
        raise ValueError("At least one split size must be positive.")
    normalized = {key: value / total for key, value in raw_values.items()}
    if normalized["train"] <= 0:
        raise ValueError("Train split must be positive for normal images.")
    if normalized["val"] + normalized["test"] <= 0:
        raise ValueError("At least one of val/test must be positive.")
    return SplitWeights(**normalized)


def split_paths(paths: list[Path], first_ratio: float, second_ratio: float) -> tuple[list[Path], list[Path]]:
    if not paths:
        return [], []
    boundary = int(round(len(paths) * first_ratio / max(first_ratio + second_ratio, 1e-12)))
    boundary = max(0, min(boundary, len(paths)))
    return paths[:boundary], paths[boundary:]


def split_normal_paths(paths: list[Path], weights: SplitWeights) -> dict[str, list[Path]]:
    total = len(paths)
    train_end = int(round(total * weights.train))
    val_end = int(round(total * (weights.train + weights.val)))
    train_end = max(0, min(train_end, total))
    val_end = max(train_end, min(val_end, total))
    return {
        "train": paths[:train_end],
        "val": paths[train_end:val_end],
        "test": paths[val_end:],
    }


def materialize_files(paths: list[Path], destination_dir: Path, copy_mode: str) -> list[str]:
    ensure_dir(destination_dir)
    materialized_names: list[str] = []
    for source_path in paths:
        destination_path = destination_dir / source_path.name
        if destination_path.exists():
            destination_path.unlink()
        if copy_mode == "hardlink":
            destination_path.hardlink_to(source_path)
        else:
            shutil.copy2(source_path, destination_path)
        materialized_names.append(source_path.name)
    return materialized_names


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)




if __name__ == "__main__":
    main()
