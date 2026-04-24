from __future__ import annotations

from threading import RLock

from app.core.config import settings
from pipeline.runtime import PipelineConfig, PipelineFrameResult, SmartTrashPipeline


class PipelineService:
    def __init__(self) -> None:
        self._pipeline: SmartTrashPipeline | None = None
        self._lock = RLock()

    def get_pipeline(self) -> SmartTrashPipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = SmartTrashPipeline(self._build_config())
        return self._pipeline

    def process_frame(self, content: bytes) -> PipelineFrameResult:
        with self._lock:
            return self.get_pipeline().process_frame(content)

    def state(self) -> dict:
        with self._lock:
            return self.get_pipeline().state()

    def reset(self) -> dict:
        with self._lock:
            pipeline = self.get_pipeline()
            pipeline.reset()
            return pipeline.state()

    @staticmethod
    def _build_config() -> PipelineConfig:
        score_threshold_override = settings.pipeline_autoencoder_threshold
        if score_threshold_override is None:
            score_threshold_override = settings.pipeline_spotter_score_threshold_override

        return PipelineConfig(
            project_root=settings.project_root,
            spotter_window_size=settings.pipeline_spotter_window_size,
            spotter_true_ratio_threshold=settings.pipeline_spotter_true_ratio_threshold,
            spotter_config_path=settings.pipeline_spotter_config,
            spotter_checkpoint_path=settings.pipeline_spotter_checkpoint,
            spotter_device=settings.pipeline_spotter_device,
            spotter_score_threshold_override=score_threshold_override,
            spotter_raw_score_threshold_override=settings.pipeline_spotter_raw_score_threshold_override,
            classifier_weights_path=settings.pipeline_classifier_weights,
            classifier_device=settings.pipeline_classifier_device,
            classifier_imgsz=settings.pipeline_classifier_imgsz,
            classifier_conf_threshold=settings.pipeline_classifier_conf_threshold,
            crop_padding=settings.pipeline_crop_padding,
            crop_min_padding=settings.pipeline_crop_min_padding,
            save_debug_artifacts=settings.pipeline_save_debug_artifacts,
            debug_dir=settings.pipeline_debug_dir,
            background_fill=settings.pipeline_background_fill,
            clean_mode=settings.pipeline_clean_mode,
            background_ref_path=settings.pipeline_background_ref,
            command_cooldown_frames=settings.pipeline_command_cooldown_frames,
        )


pipeline_service = PipelineService()
