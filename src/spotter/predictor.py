from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Iterator

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
import numpy as np
from PIL import Image
import torch

from .config import SpotterConfig, load_spotter_config


@dataclass(slots=True)
class SpotterPrediction:
    score: float | None
    label: int | None
    image_path: str | None
    anomaly_map: np.ndarray | None
    pred_mask: np.ndarray | None

    @property
    def is_anomaly(self) -> bool | None:
        if self.label is None:
            return None
        return bool(self.label)


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.squeeze(value)
    if isinstance(value, torch.Tensor):
        return np.squeeze(value.detach().cpu().numpy())
    return np.squeeze(np.asarray(value))


def _to_scalar(value: Any, cast_type: type[float] | type[int]) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flattened = value.detach().cpu().reshape(-1)
        if flattened.numel() == 0:
            return None
        return cast_type(flattened[0].item())
    if isinstance(value, np.ndarray):
        flattened = value.reshape(-1)
        if flattened.size == 0:
            return None
        return cast_type(flattened[0].item())
    return cast_type(value)


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
        evaluator=False,
        visualizer=False,
    )


def _resolve_engine_device(device: str) -> tuple[str, int]:
    if device in {"gpu", "cuda"}:
        return "gpu", 1
    if device == "cpu":
        return "cpu", 1
    return "auto", 1


def _as_uint8_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(f"Unsupported image type for prediction: {type(image)!r}")

    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]

    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0

    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


@contextmanager
def _prediction_image_path(image: str | Path | np.ndarray | torch.Tensor | Image.Image) -> Iterator[Path]:
    if isinstance(image, (str, Path)):
        yield Path(image).resolve()
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "predict.png"
        _as_uint8_image(image).save(temp_path)
        yield temp_path


class TorchSpotterPredictor:
    """Runtime wrapper that serves predictions directly from a Lightning checkpoint."""

    def __init__(self, checkpoint_path: str | Path, config: SpotterConfig, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.config = config
        self.device = device

        self.model = _build_model(config)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        accelerator, devices = _resolve_engine_device(device)
        self.engine = Engine(
            logger=False,
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=False,
            default_root_dir=self.checkpoint_path.parent.parent,
        )

    @classmethod
    def from_config_path(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path,
        workspace_root: str | Path,
        device: str = "auto",
    ) -> "TorchSpotterPredictor":
        config = load_spotter_config(config_path, workspace_root=workspace_root)
        return cls(checkpoint_path=checkpoint_path, config=config, device=device)

    def predict(self, image: str | Path | np.ndarray | torch.Tensor | Image.Image) -> SpotterPrediction:
        with _prediction_image_path(image) as image_path:
            dataset = PredictDataset(path=image_path, image_size=self.config.model.image_size)
            predictions = self.engine.predict(
                model=self.model,
                dataset=dataset,
                return_predictions=True,
            )

        batch = predictions[0]
        image_path_value = getattr(batch, "image_path", None)
        if isinstance(image_path_value, (list, tuple)):
            image_path_value = image_path_value[0] if image_path_value else None
        if image_path_value is not None:
            image_path_value = str(image_path_value)

        return SpotterPrediction(
            score=_to_scalar(getattr(batch, "pred_score", None), float),
            label=_to_scalar(getattr(batch, "pred_label", None), int),
            image_path=image_path_value,
            anomaly_map=_to_numpy(getattr(batch, "anomaly_map", None)),
            pred_mask=_to_numpy(getattr(batch, "pred_mask", None)),
        )
