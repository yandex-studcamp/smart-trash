from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anomalib.data import Folder

from ..config import AnomalibSpotterConfig


@dataclass(slots=True)
class AnomalibPreparedDatasetArtifact:
    exp_name: str
    dataset_root: Path
    manifest_path: Path
    normal_train_count: int
    normal_test_count: int
    anomaly_test_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_name": self.exp_name,
            "dataset_root": str(self.dataset_root),
            "manifest_path": str(self.manifest_path),
            "normal_train_count": self.normal_train_count,
            "normal_test_count": self.normal_test_count,
            "anomaly_test_count": self.anomaly_test_count,
        }


def _collect_image_files(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    ]
    if not files:
        raise FileNotFoundError(f"No image files with extensions {extensions} found in {directory}.")
    return files


def _copy_or_link(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _ensure_clean_dir(directory: Path, force: bool) -> None:
    if directory.exists():
        if not force:
            raise FileExistsError(
                f"Prepared dataset already exists at {directory}. "
                "Use --force if you want to recreate it."
            )
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def anomalib_dataset_is_prepared(config: AnomalibSpotterConfig, exp_name: str) -> bool:
    dataset_root = config.dataset_root_for(exp_name)
    required_dirs = [
        dataset_root / config.dataset.train_good_dir,
        dataset_root / config.dataset.test_good_dir,
        dataset_root / config.dataset.test_anomaly_dir,
    ]
    return all(path.exists() and any(path.iterdir()) for path in required_dirs)


def build_anomalib_datamodule(
    config: AnomalibSpotterConfig,
    dataset_root: Path,
    exp_name: str,
) -> Folder:
    return Folder(
        name=exp_name,
        root=dataset_root,
        normal_dir=config.dataset.train_good_dir,
        abnormal_dir=config.dataset.test_anomaly_dir,
        normal_test_dir=config.dataset.test_good_dir,
        train_batch_size=config.model.train_batch_size,
        eval_batch_size=config.model.eval_batch_size,
        num_workers=config.model.num_workers,
        test_split_mode=config.dataset.test_split_mode,
        test_split_ratio=config.dataset.test_split_ratio,
        val_split_mode=config.dataset.val_split_mode,
        val_split_ratio=config.dataset.val_split_ratio,
        seed=config.seed,
    )


def collect_anomalib_test_samples(
    config: AnomalibSpotterConfig,
    exp_name: str,
) -> list[tuple[Path, int, str]]:
    dataset_root = config.dataset_root_for(exp_name)
    normal_dir = dataset_root / config.dataset.test_good_dir
    anomaly_dir = dataset_root / config.dataset.test_anomaly_dir

    samples: list[tuple[Path, int, str]] = []
    for path in sorted(normal_dir.iterdir()):
        if path.is_file():
            samples.append((path, 0, "good"))
    for path in sorted(anomaly_dir.iterdir()):
        if path.is_file():
            samples.append((path, 1, "anomaly"))
    return samples


def prepare_anomalib_spotter_dataset(
    config: AnomalibSpotterConfig,
    exp_name: str,
    *,
    force: bool = False,
) -> AnomalibPreparedDatasetArtifact:
    normal_files = _collect_image_files(config.raw_data.normal_dir, config.raw_data.extensions)
    anomaly_files = _collect_image_files(config.raw_data.anomaly_dir, config.raw_data.extensions)

    dataset_root = config.dataset_root_for(exp_name)
    _ensure_clean_dir(dataset_root, force=force)

    rng = random.Random(config.seed)
    normal_files = normal_files.copy()
    rng.shuffle(normal_files)

    split_index = int(round(len(normal_files) * config.dataset.train_ratio))
    split_index = max(1, min(split_index, len(normal_files) - 1))

    train_good = normal_files[:split_index]
    test_good = normal_files[split_index:]
    test_anomaly = anomaly_files

    train_dir = dataset_root / config.dataset.train_good_dir
    test_good_dir = dataset_root / config.dataset.test_good_dir
    test_anomaly_dir = dataset_root / config.dataset.test_anomaly_dir

    for source in train_good:
        _copy_or_link(source, train_dir / source.name, config.dataset.copy_mode)
    for source in test_good:
        _copy_or_link(source, test_good_dir / source.name, config.dataset.copy_mode)
    for source in test_anomaly:
        _copy_or_link(source, test_anomaly_dir / source.name, config.dataset.copy_mode)

    manifest = {
        "exp_name": exp_name,
        "seed": config.seed,
        "copy_mode": config.dataset.copy_mode,
        "normal_train_count": len(train_good),
        "normal_test_count": len(test_good),
        "anomaly_test_count": len(test_anomaly),
        "train_good_dir": str(train_dir),
        "test_good_dir": str(test_good_dir),
        "test_anomaly_dir": str(test_anomaly_dir),
        "train_good_files": [path.name for path in train_good],
        "test_good_files": [path.name for path in test_good],
        "test_anomaly_files": [path.name for path in test_anomaly],
    }
    manifest_path = dataset_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return AnomalibPreparedDatasetArtifact(
        exp_name=exp_name,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        normal_train_count=len(train_good),
        normal_test_count=len(test_good),
        anomaly_test_count=len(test_anomaly),
    )
