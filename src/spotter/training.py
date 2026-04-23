from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from anomalib.data import Folder
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning import seed_everything
import torch

from .config import SpotterConfig
from .dataset import PreparedDatasetArtifact, dataset_is_prepared, prepare_spotter_dataset


@dataclass(slots=True)
class TrainingArtifact:
    exp_name: str
    dataset_root: Path
    run_root: Path
    checkpoint_path: Path
    summary_path: Path
    metrics: list[dict[str, Any]]
    export_path: Path | None
    prepared_dataset: PreparedDatasetArtifact | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_name": self.exp_name,
            "dataset_root": str(self.dataset_root),
            "run_root": str(self.run_root),
            "checkpoint_path": str(self.checkpoint_path),
            "summary_path": str(self.summary_path),
            "metrics": self.metrics,
            "export_path": None if self.export_path is None else str(self.export_path),
            "prepared_dataset": None if self.prepared_dataset is None else self.prepared_dataset.to_dict(),
        }


def _resolve_accelerator(accelerator: str) -> str:
    if accelerator != "auto":
        return accelerator
    return "gpu" if torch.cuda.is_available() else "cpu"


def _resolve_devices(accelerator: str, devices: int | str | list[int]) -> int | str | list[int]:
    if accelerator == "cpu" and devices == "auto":
        return 1
    return devices


def _build_datamodule(config: SpotterConfig, dataset_root: Path, exp_name: str) -> Folder:
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


def _build_model(config: SpotterConfig) -> Patchcore:
    pre_processor = Patchcore.configure_pre_processor(
        image_size=config.model.image_size,
        center_crop_size=config.model.center_crop_size,
    )
    return Patchcore(
        backbone=config.model.backbone,
        layers=config.model.layers,
        coreset_sampling_ratio=config.model.coreset_sampling_ratio,
        num_neighbors=config.model.num_neighbors,
        precision=config.model.precision,
        pre_processor=pre_processor,
    )


def _build_engine(config: SpotterConfig, run_root: Path) -> Engine:
    accelerator = _resolve_accelerator(config.engine.accelerator)
    devices = _resolve_devices(accelerator, config.engine.devices)
    return Engine(
        default_root_dir=run_root,
        logger=False,
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.engine.max_epochs,
        deterministic=config.engine.deterministic,
        precision=config.engine.trainer_precision,
    )


def _save_summary(
    config: SpotterConfig,
    artifact: TrainingArtifact,
) -> None:
    artifact.run_root.mkdir(parents=True, exist_ok=True)
    resolved_config_path = artifact.run_root / "resolved_config.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")

    artifact.summary_path.write_text(
        json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _export_model_if_requested(
    config: SpotterConfig,
    engine: Engine,
    model: Patchcore,
    datamodule: Folder,
    run_root: Path,
    checkpoint_path: Path,
) -> Path | None:
    if config.engine.export_type == "none":
        return None

    export_root = config.engine.export_root or (run_root / "export")
    export_root.mkdir(parents=True, exist_ok=True)
    return engine.export(
        model=model,
        export_type=ExportType(config.engine.export_type),
        export_root=export_root,
        datamodule=datamodule,
        ckpt_path=checkpoint_path,
    )


def train_patchcore_experiment(
    config: SpotterConfig,
    exp_name: str,
    *,
    prepare_if_missing: bool = True,
    force_prepare: bool = False,
) -> TrainingArtifact:
    prepared_dataset: PreparedDatasetArtifact | None = None
    if prepare_if_missing and (force_prepare or not dataset_is_prepared(config, exp_name)):
        prepared_dataset = prepare_spotter_dataset(config, exp_name, force=force_prepare)

    dataset_root = config.dataset_root_for(exp_name)
    if not dataset_is_prepared(config, exp_name):
        raise FileNotFoundError(
            f"Prepared dataset for experiment '{exp_name}' was not found in {dataset_root}. "
            "Run the prepare script first or enable prepare_if_missing."
        )

    run_root = config.run_root_for(exp_name)
    checkpoint_path = config.checkpoint_path_for(exp_name)
    summary_path = run_root / "train_summary.json"

    seed_everything(config.seed, workers=True)

    datamodule = _build_datamodule(config, dataset_root, exp_name)
    model = _build_model(config)
    engine = _build_engine(config, run_root)

    engine.fit(model=model, datamodule=datamodule)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    engine.trainer.save_checkpoint(str(checkpoint_path))

    metrics = [dict(item) for item in engine.test(model=model, datamodule=datamodule)]
    export_path = _export_model_if_requested(
        config=config,
        engine=engine,
        model=model,
        datamodule=datamodule,
        run_root=run_root,
        checkpoint_path=checkpoint_path,
    )

    artifact = TrainingArtifact(
        exp_name=exp_name,
        dataset_root=dataset_root,
        run_root=run_root,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        metrics=metrics,
        export_path=export_path,
        prepared_dataset=prepared_dataset,
    )
    _save_summary(config, artifact)
    return artifact
