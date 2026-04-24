from __future__ import annotations

from pathlib import Path

from anomalib.models import Patchcore
import torch

from ..config import AnomalibSpotterConfig


def build_anomalib_patchcore_model(
    config: AnomalibSpotterConfig,
    *,
    evaluator: bool = True,
    visualizer: bool = False,
) -> Patchcore:
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
        evaluator=evaluator,
        visualizer=visualizer,
    )


def load_anomalib_patchcore_weights(model: Patchcore, checkpoint_path: str | Path) -> dict:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint


def extract_anomalib_image_thresholds(model: Patchcore) -> tuple[float | None, float | None]:
    post_processor = model.post_processor
    raw_threshold = getattr(post_processor, "image_threshold", None)
    normalized_threshold = getattr(post_processor, "normalized_image_threshold", None)

    raw_threshold_value = None
    normalized_threshold_value = None

    if isinstance(raw_threshold, torch.Tensor) and not torch.isnan(raw_threshold):
        raw_threshold_value = float(raw_threshold.item())
    elif raw_threshold is not None:
        raw_threshold_value = float(raw_threshold)

    if isinstance(normalized_threshold, torch.Tensor) and not torch.isnan(normalized_threshold):
        normalized_threshold_value = float(normalized_threshold.item())
    elif normalized_threshold is not None:
        normalized_threshold_value = float(normalized_threshold)

    return raw_threshold_value, normalized_threshold_value
