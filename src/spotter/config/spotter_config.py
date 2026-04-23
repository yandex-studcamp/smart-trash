from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils.spotter_utils import load_json, save_json


@dataclass
class SpotterDataConfig:
    train_normal_dir: str = "data/spotter/prepared/esp_patchcore/train/good"
    val_normal_dir: str | None = "data/spotter/prepared/esp_patchcore/test/good"
    eval_normal_dir: str | None = "data/spotter/prepared/esp_patchcore/test/good"
    eval_anomaly_dir: str | None = "data/spotter/prepared/esp_patchcore/test/anomaly"
    image_size: list[int] = field(default_factory=lambda: [256, 256])
    train_batch_size: int = 8
    eval_batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    val_split: float = 0.2
    max_train_images: int | None = None
    max_eval_normal_images: int | None = None
    max_eval_anomaly_images: int | None = None
    image_extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    )


@dataclass
class SpotterAugmentationConfig:
    noise_probability: float = 0.9
    noise_std_min: float = 0.01
    noise_std_max: float = 0.08
    brightness_probability: float = 0.6
    brightness_min: float = 0.75
    brightness_max: float = 1.20
    shadow_probability: float = 0.5
    shadow_strength_min: float = 0.25
    shadow_strength_max: float = 0.65
    shadow_box_scale_min: float = 0.15
    shadow_box_scale_max: float = 0.50
    cutout_probability: float = 0.7
    cutout_holes_min: int = 1
    cutout_holes_max: int = 3
    cutout_scale_min: float = 0.08
    cutout_scale_max: float = 0.22


@dataclass
class SpotterModelConfig:
    in_channels: int = 3
    base_channels: int = 64
    attention_reduction: int = 16


@dataclass
class SpotterTrainingConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mse_weight: float = 0.6
    ssim_weight: float = 0.4
    grad_clip_norm: float | None = 1.0
    save_every_n_epochs: int = 5
    sample_visualizations: int = 4


@dataclass
class SpotterInferenceConfig:
    image_score_reduction: str = "max"
    image_score_quantile: float = 0.995
    pixel_threshold: float = 0.12
    blur_kernel_size: int = 5
    morph_kernel_size: int = 5
    min_contour_area: int = 80
    max_visualizations_per_split: int = 6


@dataclass
class SpotterConfig:
    exp_name: str = "spotter_daae"
    seed: int = 42
    device: str = "auto"
    data: SpotterDataConfig = field(default_factory=SpotterDataConfig)
    augmentation: SpotterAugmentationConfig = field(default_factory=SpotterAugmentationConfig)
    model: SpotterModelConfig = field(default_factory=SpotterModelConfig)
    training: SpotterTrainingConfig = field(default_factory=SpotterTrainingConfig)
    inference: SpotterInferenceConfig = field(default_factory=SpotterInferenceConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpotterConfig":
        return cls(
            exp_name=payload.get("exp_name", "spotter_daae"),
            seed=payload.get("seed", 42),
            device=payload.get("device", "auto"),
            data=SpotterDataConfig(**payload.get("data", {})),
            augmentation=SpotterAugmentationConfig(**payload.get("augmentation", {})),
            model=SpotterModelConfig(**payload.get("model", {})),
            training=SpotterTrainingConfig(**payload.get("training", {})),
            inference=SpotterInferenceConfig(**payload.get("inference", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_spotter_config(path: str | Path) -> SpotterConfig:
    return SpotterConfig.from_dict(load_json(path))


def save_spotter_config(config: SpotterConfig, path: str | Path) -> None:
    save_json(config.to_dict(), path)
