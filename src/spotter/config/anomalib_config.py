from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "raw_data": {
        "normal_dir": "data/raw/captures_esp/normal",
        "anomaly_dir": "data/raw/captures_esp/anomaly",
        "extensions": [".jpg", ".jpeg", ".png", ".bmp"],
    },
    "dataset": {
        "prepared_root": "data/spotter/prepared",
        "train_ratio": 0.8,
        "copy_mode": "copy",
        "train_good_dir": "train/good",
        "test_good_dir": "test/good",
        "test_anomaly_dir": "test/anomaly",
        "test_split_mode": "from_dir",
        "test_split_ratio": 0.2,
        "val_split_mode": "from_test",
        "val_split_ratio": 0.5,
    },
    "model": {
        "image_size": [256, 256],
        "center_crop_size": [224, 224],
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_workers": 15,
        "precision": "float32",
    },
    "engine": {
        "results_root": "experiments/spotter",
        "accelerator": "auto",
        "devices": 1,
        "max_epochs": 1,
        "trainer_precision": "32-true",
        "deterministic": True,
        "export_type": "none",
        "export_root": None,
    },
}


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _resolve_path(path_like: str | Path | None, workspace_root: Path) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def _to_int_pair(value: list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"Expected a pair of integers, got {value!r}.")
    return int(value[0]), int(value[1])


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class AnomalibRawDataConfig:
    normal_dir: Path
    anomaly_dir: Path
    extensions: tuple[str, ...]


@dataclass(slots=True)
class AnomalibDatasetConfig:
    prepared_root: Path
    train_ratio: float
    copy_mode: str
    train_good_dir: str
    test_good_dir: str
    test_anomaly_dir: str
    test_split_mode: str
    test_split_ratio: float
    val_split_mode: str
    val_split_ratio: float


@dataclass(slots=True)
class AnomalibModelConfig:
    image_size: tuple[int, int]
    center_crop_size: tuple[int, int] | None
    backbone: str
    layers: tuple[str, ...]
    coreset_sampling_ratio: float
    num_neighbors: int
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    precision: str


@dataclass(slots=True)
class AnomalibEngineConfig:
    results_root: Path
    accelerator: str
    devices: int | str | list[int]
    max_epochs: int
    trainer_precision: str
    deterministic: bool
    export_type: str
    export_root: Path | None


@dataclass(slots=True)
class AnomalibSpotterConfig:
    seed: int
    raw_data: AnomalibRawDataConfig
    dataset: AnomalibDatasetConfig
    model: AnomalibModelConfig
    engine: AnomalibEngineConfig

    def dataset_root_for(self, exp_name: str) -> Path:
        return (self.dataset.prepared_root / exp_name).resolve()

    def run_root_for(self, exp_name: str) -> Path:
        return (self.engine.results_root / exp_name).resolve()

    def checkpoint_path_for(self, exp_name: str) -> Path:
        return self.run_root_for(exp_name) / "weights" / "patchcore.ckpt"

    def evaluation_root_for(self, exp_name: str) -> Path:
        return self.run_root_for(exp_name) / "evaluation"

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(asdict(self))


def load_anomalib_spotter_config(
    config_path: str | Path | None,
    workspace_root: str | Path,
) -> AnomalibSpotterConfig:
    workspace_root = Path(workspace_root).resolve()
    merged = deepcopy(DEFAULT_CONFIG)

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = workspace_root / config_path
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Spotter config must be a YAML mapping.")
        merged = _deep_update(merged, loaded)

    config = AnomalibSpotterConfig(
        seed=int(merged["seed"]),
        raw_data=AnomalibRawDataConfig(
            normal_dir=_resolve_path(merged["raw_data"]["normal_dir"], workspace_root),
            anomaly_dir=_resolve_path(merged["raw_data"]["anomaly_dir"], workspace_root),
            extensions=tuple(str(item).lower() for item in merged["raw_data"]["extensions"]),
        ),
        dataset=AnomalibDatasetConfig(
            prepared_root=_resolve_path(merged["dataset"]["prepared_root"], workspace_root),
            train_ratio=float(merged["dataset"]["train_ratio"]),
            copy_mode=str(merged["dataset"]["copy_mode"]).lower(),
            train_good_dir=str(merged["dataset"]["train_good_dir"]),
            test_good_dir=str(merged["dataset"]["test_good_dir"]),
            test_anomaly_dir=str(merged["dataset"]["test_anomaly_dir"]),
            test_split_mode=str(merged["dataset"]["test_split_mode"]),
            test_split_ratio=float(merged["dataset"]["test_split_ratio"]),
            val_split_mode=str(merged["dataset"]["val_split_mode"]),
            val_split_ratio=float(merged["dataset"]["val_split_ratio"]),
        ),
        model=AnomalibModelConfig(
            image_size=_to_int_pair(merged["model"]["image_size"]),
            center_crop_size=_to_int_pair(merged["model"]["center_crop_size"]),
            backbone=str(merged["model"]["backbone"]),
            layers=tuple(str(item) for item in merged["model"]["layers"]),
            coreset_sampling_ratio=float(merged["model"]["coreset_sampling_ratio"]),
            num_neighbors=int(merged["model"]["num_neighbors"]),
            train_batch_size=int(merged["model"]["train_batch_size"]),
            eval_batch_size=int(merged["model"]["eval_batch_size"]),
            num_workers=int(merged["model"]["num_workers"]),
            precision=str(merged["model"]["precision"]),
        ),
        engine=AnomalibEngineConfig(
            results_root=_resolve_path(merged["engine"]["results_root"], workspace_root),
            accelerator=str(merged["engine"]["accelerator"]),
            devices=merged["engine"]["devices"],
            max_epochs=int(merged["engine"]["max_epochs"]),
            trainer_precision=str(merged["engine"]["trainer_precision"]),
            deterministic=bool(merged["engine"]["deterministic"]),
            export_type=str(merged["engine"]["export_type"]).lower(),
            export_root=_resolve_path(merged["engine"]["export_root"], workspace_root),
        ),
    )

    if not 0.0 < config.dataset.train_ratio < 1.0:
        raise ValueError("dataset.train_ratio must be between 0 and 1.")
    if config.dataset.copy_mode not in {"copy", "hardlink"}:
        raise ValueError("dataset.copy_mode must be either 'copy' or 'hardlink'.")
    if config.engine.export_type not in {"none", "onnx", "openvino", "torch"}:
        raise ValueError("engine.export_type must be one of: none, onnx, openvino, torch.")
    return config
