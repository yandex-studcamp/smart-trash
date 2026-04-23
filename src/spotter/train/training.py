from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from lightning import seed_everything
import torch

from ..config import SpotterConfig
from ..data import PreparedDatasetArtifact, build_folder_datamodule, dataset_is_prepared, prepare_spotter_dataset
from ..models import build_patchcore_model


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


def build_spotter_engine(config: SpotterConfig, run_root: Path) -> Engine:
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
        enable_progress_bar=False,
    )


def _save_summary(config: SpotterConfig, artifact: TrainingArtifact) -> None:
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
    model,
    datamodule,
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

    datamodule = build_folder_datamodule(config, dataset_root, exp_name)
    model = build_patchcore_model(config, evaluator=True, visualizer=False)
    engine = build_spotter_engine(config, run_root)

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
