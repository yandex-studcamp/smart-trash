from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ExperimentPaths:
    root_dir: Path
    weights_dir: Path
    artifacts_dir: Path
    config_path: Path


def resolve_repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def make_experiment_paths(exp_name: str) -> ExperimentPaths:
    root_dir = ensure_dir(REPO_ROOT / "experiments" / "spotter" / exp_name)
    weights_dir = ensure_dir(root_dir / "weights")
    artifacts_dir = ensure_dir(root_dir / "artifacts")
    return ExperimentPaths(
        root_dir=root_dir,
        weights_dir=weights_dir,
        artifacts_dir=artifacts_dir,
        config_path=root_dir / "config.json",
    )


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"cudaGetDeviceCount\(\) returned cudaErrorNotSupported.*",
        )
        has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"cudaGetDeviceCount\(\) returned cudaErrorNotSupported.*",
            )
            has_cuda = torch.cuda.is_available()
        return torch.device("cuda" if has_cuda else "cpu")
    return torch.device(requested_device)


def collect_image_files(
    paths: Iterable[str | Path],
    extensions: Sequence[str] | None = None,
) -> list[Path]:
    allowed_extensions = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (extensions or DEFAULT_IMAGE_EXTENSIONS)
    )
    image_files: list[Path] = []
    for raw_path in paths:
        resolved_path = resolve_repo_path(raw_path)
        if resolved_path is None:
            continue
        if not resolved_path.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved_path}")
        if resolved_path.is_file():
            if resolved_path.suffix.lower() in allowed_extensions:
                image_files.append(resolved_path)
            continue
        image_files.extend(
            sorted(
                file_path
                for file_path in resolved_path.rglob("*")
                if file_path.is_file() and file_path.suffix.lower() in allowed_extensions
            )
        )
    return sorted(dict.fromkeys(image_files))
