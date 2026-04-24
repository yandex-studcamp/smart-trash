from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from ..config.autoencoder_config import AutoencoderSpotterConfig, load_autoencoder_spotter_config
from ..utils.spotter_utils import load_json, make_experiment_paths, select_device
from .autoencoder_inference import (
    compute_residual_map,
    extract_residual_boxes,
    load_autoencoder_checkpoint_model,
    reduce_anomaly_score,
    tensor_to_rgb_image,
)
from .base_spotter import BaseSpotter, SpotterImageInput


@dataclass(frozen=True)
class AutoencoderSpotterPrediction:
    is_anomaly: bool
    score: float
    threshold: float
    image_path: str | None
    residual_max: float
    residual_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "score": self.score,
            "threshold": self.threshold,
            "image_path": self.image_path,
            "residual_max": self.residual_max,
            "residual_mean": self.residual_mean,
        }


class AutoencoderSpotter(BaseSpotter):
    def __init__(
        self,
        config_path: str | Path,
        weights_path: str | Path,
        threshold: float | None = None,
        calibration_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.weights_path = Path(weights_path)
        self.config = self._load_config()
        if device is not None:
            self.config.device = device
        self.device = select_device(self.config.device)
        self.model = load_autoencoder_checkpoint_model(
            config=self.config,
            weights_path=self.weights_path,
            device=self.device,
        )
        self.threshold = self._resolve_threshold(
            threshold=threshold,
            calibration_path=Path(calibration_path) if calibration_path is not None else None,
        )

    @classmethod
    def from_experiment(
        cls,
        exp_name: str,
        device: str | None = None,
        threshold: float | None = None,
    ) -> "AutoencoderSpotter":
        experiment_paths = make_experiment_paths(exp_name)
        config_path = experiment_paths.config_path
        weights_path = experiment_paths.weights_dir / "best_autoencoder_spotter.pt"
        calibration_path = experiment_paths.artifacts_dir / "calibration.json"
        return cls(
            config_path=config_path,
            weights_path=weights_path,
            threshold=threshold,
            calibration_path=calibration_path,
            device=device,
        )

    def predict(self, image: SpotterImageInput) -> bool:
        return bool(self.predict_details(image)["is_anomaly"])

    def predict_details(self, image: SpotterImageInput) -> dict[str, Any]:
        image_path = str(image) if isinstance(image, (str, Path)) else None
        input_tensor = self._prepare_image(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructed = self.model(input_tensor)[0].cpu()

        original = input_tensor[0].cpu()
        residual_map = compute_residual_map(original, reconstructed)
        score = reduce_anomaly_score(
            residual_map=residual_map,
            reduction=self.config.inference.image_score_reduction,
            quantile=self.config.inference.image_score_quantile,
        )
        boxes, binary_mask = extract_residual_boxes(
            residual_map=residual_map,
            pixel_threshold=self.config.inference.pixel_threshold,
            blur_kernel_size=self.config.inference.blur_kernel_size,
            morph_kernel_size=self.config.inference.morph_kernel_size,
            min_contour_area=self.config.inference.min_contour_area,
        )
        prediction = AutoencoderSpotterPrediction(
            is_anomaly=bool(score >= self.threshold),
            score=float(score),
            threshold=float(self.threshold),
            image_path=image_path,
            residual_max=float(residual_map.max().item()),
            residual_mean=float(residual_map.mean().item()),
        )
        result = prediction.to_dict()
        result["boxes"] = boxes
        result["box_count"] = len(boxes)
        result["original_tensor"] = original
        result["reconstructed_tensor"] = reconstructed
        result["residual_map"] = residual_map
        result["binary_mask"] = binary_mask
        result["original_image"] = tensor_to_rgb_image(original)
        result["reconstructed_image"] = tensor_to_rgb_image(reconstructed)
        return result

    def _resolve_threshold(
        self,
        threshold: float | None,
        calibration_path: Path | None,
    ) -> float:
        if threshold is not None:
            return float(threshold)
        if calibration_path is not None and calibration_path.exists():
            calibration_payload = load_json(calibration_path)
            if "best_threshold" in calibration_payload:
                return float(calibration_payload["best_threshold"])
        raise FileNotFoundError(
            "Threshold is not provided and calibration.json with best_threshold was not found."
        )

    def _load_config(self) -> AutoencoderSpotterConfig:
        if self.config_path.exists():
            return load_autoencoder_spotter_config(self.config_path)
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        checkpoint_config = checkpoint.get("config")
        if checkpoint_config is None:
            raise FileNotFoundError(
                f"Config file was not found at {self.config_path} and checkpoint does not contain config."
            )
        return AutoencoderSpotterConfig.from_dict(checkpoint_config)

    def _prepare_image(
        self,
        image: SpotterImageInput,
    ) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image_tensor = image.detach().cpu().float()
            if image_tensor.ndim != 3:
                raise ValueError("Expected image tensor with shape [C, H, W].")
            if image_tensor.max().item() > 1.0:
                image_tensor = image_tensor / 255.0
            pil_image = TF.to_pil_image(image_tensor.clamp(0.0, 1.0))
        elif isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            array = image
            if array.ndim != 3:
                raise ValueError("Expected image array with shape [H, W, C].")
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array).convert("RGB")
        else:
            raise TypeError(
                "Unsupported image type. Expected path, PIL.Image, numpy.ndarray, or torch.Tensor."
            )

        resized = pil_image.resize(
            (self.config.data.image_size[1], self.config.data.image_size[0]),
            Image.Resampling.BILINEAR,
        )
        return TF.to_tensor(resized)
