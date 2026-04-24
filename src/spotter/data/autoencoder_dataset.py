from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from ..config.autoencoder_config import AutoencoderAugmentationConfig
from ..utils.spotter_utils import collect_image_files


def load_autoencoder_image_tensor(image_path: str | Path, image_size: list[int]) -> torch.Tensor:
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    resized = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    return TF.to_tensor(resized)


def build_autoencoder_train_val_image_lists(
    train_normal_dir: str,
    val_normal_dir: str | None,
    image_extensions: list[str],
    val_split: float,
    seed: int,
    max_train_images: int | None = None,
) -> tuple[list[Path], list[Path]]:
    train_paths = collect_image_files([train_normal_dir], image_extensions)
    if max_train_images is not None:
        train_paths = train_paths[:max_train_images]
    if not train_paths:
        raise ValueError("No training images were found for the spotter.")

    if val_normal_dir:
        val_paths = collect_image_files([val_normal_dir], image_extensions)
        if not val_paths:
            raise ValueError("Validation directory is configured but contains no images.")
        return train_paths, val_paths

    if len(train_paths) == 1:
        return train_paths, train_paths

    rng = random.Random(seed)
    shuffled_paths = train_paths.copy()
    rng.shuffle(shuffled_paths)
    val_count = max(1, int(round(len(shuffled_paths) * val_split))) if val_split > 0 else 1
    val_count = min(val_count, len(shuffled_paths) - 1)
    return shuffled_paths[val_count:], shuffled_paths[:val_count]


def build_autoencoder_eval_records(
    normal_dir: str | None,
    anomaly_dir: str | None,
    image_extensions: list[str],
    max_eval_normal_images: int | None = None,
    max_eval_anomaly_images: int | None = None,
) -> list[tuple[Path, int]]:
    records: list[tuple[Path, int]] = []
    if normal_dir:
        normal_paths = collect_image_files([normal_dir], image_extensions)
        if max_eval_normal_images is not None:
            normal_paths = normal_paths[:max_eval_normal_images]
        records.extend((path, 0) for path in normal_paths)
    if anomaly_dir:
        anomaly_paths = collect_image_files([anomaly_dir], image_extensions)
        if max_eval_anomaly_images is not None:
            anomaly_paths = anomaly_paths[:max_eval_anomaly_images]
        records.extend((path, 1) for path in anomaly_paths)
    if not records:
        raise ValueError("No evaluation images were found for the spotter.")
    return records


class AutoencoderTrainingDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        image_size: list[int],
        augmentation: AutoencoderAugmentationConfig,
        enable_corruption: bool,
    ) -> None:
        self.image_paths = image_paths
        self.image_size = image_size
        self.augmentation = augmentation
        self.enable_corruption = enable_corruption

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        clean_image = load_autoencoder_image_tensor(image_path, self.image_size)
        corrupted_image = self._apply_corruption(clean_image) if self.enable_corruption else clean_image.clone()
        return {
            "input": corrupted_image,
            "target": clean_image,
            "path": str(image_path),
        }

    def _apply_corruption(self, clean_image: torch.Tensor) -> torch.Tensor:
        corrupted = clean_image.clone()
        corrupted = self._apply_gaussian_noise(corrupted)
        corrupted = self._apply_brightness_shift(corrupted)
        corrupted = self._apply_shadow_patch(corrupted)
        corrupted = self._apply_cutout(corrupted)
        return corrupted.clamp_(0.0, 1.0)

    def _apply_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.augmentation.noise_probability:
            return image
        noise_std = random.uniform(
            self.augmentation.noise_std_min,
            self.augmentation.noise_std_max,
        )
        return image + torch.randn_like(image) * noise_std

    def _apply_brightness_shift(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.augmentation.brightness_probability:
            return image
        brightness_factor = random.uniform(
            self.augmentation.brightness_min,
            self.augmentation.brightness_max,
        )
        return image * brightness_factor

    def _apply_shadow_patch(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.augmentation.shadow_probability:
            return image

        _, height, width = image.shape
        patch_height = max(1, int(height * random.uniform(
            self.augmentation.shadow_box_scale_min,
            self.augmentation.shadow_box_scale_max,
        )))
        patch_width = max(1, int(width * random.uniform(
            self.augmentation.shadow_box_scale_min,
            self.augmentation.shadow_box_scale_max,
        )))
        top = random.randint(0, max(0, height - patch_height))
        left = random.randint(0, max(0, width - patch_width))
        darkness = random.uniform(
            self.augmentation.shadow_strength_min,
            self.augmentation.shadow_strength_max,
        )
        image[:, top : top + patch_height, left : left + patch_width] *= darkness
        return image

    def _apply_cutout(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.augmentation.cutout_probability:
            return image

        _, height, width = image.shape
        holes = random.randint(
            self.augmentation.cutout_holes_min,
            self.augmentation.cutout_holes_max,
        )
        for _ in range(holes):
            patch_height = max(1, int(height * random.uniform(
                self.augmentation.cutout_scale_min,
                self.augmentation.cutout_scale_max,
            )))
            patch_width = max(1, int(width * random.uniform(
                self.augmentation.cutout_scale_min,
                self.augmentation.cutout_scale_max,
            )))
            top = random.randint(0, max(0, height - patch_height))
            left = random.randint(0, max(0, width - patch_width))
            image[:, top : top + patch_height, left : left + patch_width] = 0.0
        return image


class AutoencoderEvaluationDataset(Dataset):
    def __init__(self, records: list[tuple[Path, int]], image_size: list[int]) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        image_path, label = self.records[index]
        image_tensor = load_autoencoder_image_tensor(image_path, self.image_size)
        return {
            "image": image_tensor,
            "label": label,
            "path": str(image_path),
        }
