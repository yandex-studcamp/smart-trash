from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from pipeline.classifier import COMMAND_LABELS, YoloMaterialClassifier, command_to_label
from pipeline.crop import (
    CropInfo,
    build_clean_object_mask,
    crop_from_spotter_maps,
    get_prediction_score,
    get_prediction_threshold,
    heatmap_rgb,
    item_present_from_prediction,
    make_clean_crop,
    resize_heatmap,
)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    project_root: Path
    spotter_window_size: int
    spotter_true_ratio_threshold: float
    spotter_config_path: Path
    spotter_checkpoint_path: Path
    spotter_device: str
    classifier_weights_path: Path
    classifier_device: str | None
    classifier_imgsz: int
    classifier_conf_threshold: float
    crop_padding: float
    crop_min_padding: int
    save_debug_artifacts: bool
    debug_dir: Path
    background_fill: str
    clean_mode: str
    background_ref_path: Path | None = None
    command_cooldown_frames: int = 0
    background_threshold_min: float = 18.0
    background_threshold_max: float = 58.0
    anchor_dilate_ratio: float = 0.18
    align_background: bool = True
    use_edge_support: bool = True


@dataclass(slots=True)
class FrameRecord:
    request_id: str
    timestamp: str
    image: Image.Image
    prediction: Any
    is_anomaly: bool
    frame_path: Path | None


@dataclass(slots=True)
class PipelineFrameResult:
    status: str
    command: int
    label: str
    spotter: dict[str, Any]
    classifier: dict[str, Any]
    artifacts: dict[str, Any]
    duplicate: bool = False
    suppressed_command: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.suppressed_command is None:
            payload.pop("suppressed_command", None)
        return payload


class SlidingWindow:
    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("Sliding window size must be greater than zero.")
        self.maxlen = int(maxlen)
        self._items: deque[FrameRecord] = deque(maxlen=self.maxlen)

    def append(self, item: FrameRecord) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterable[FrameRecord]:
        return iter(self._items)

    @property
    def items(self) -> list[FrameRecord]:
        return list(self._items)

    @property
    def true_count(self) -> int:
        return sum(1 for item in self._items if item.is_anomaly)

    @property
    def true_ratio(self) -> float:
        if not self._items:
            return 0.0
        return float(self.true_count) / float(len(self._items))

    @property
    def is_full(self) -> bool:
        return len(self._items) == self.maxlen


class SmartTrashPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.window = SlidingWindow(config.spotter_window_size)
        self._spotter: Any | None = None
        self._classifier: YoloMaterialClassifier | None = None
        self._background: Image.Image | None = None
        self._last_command: int | None = None
        self._frames_since_command = config.command_cooldown_frames + 1

    @property
    def spotter_loaded(self) -> bool:
        return self._spotter is not None

    @property
    def classifier_loaded(self) -> bool:
        return self._classifier is not None

    def reset(self) -> None:
        self.window.clear()
        self._last_command = None
        self._frames_since_command = self.config.command_cooldown_frames + 1

    def state(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "models": {
                "spotter_loaded": self.spotter_loaded,
                "classifier_loaded": self.classifier_loaded,
                "background_loaded": self._background is not None,
            },
            "spotter": {
                "window_size": self.window.maxlen,
                "filled": len(self.window),
                "true_count": self.window.true_count,
                "true_ratio": self.window.true_ratio,
                "threshold": self.config.spotter_true_ratio_threshold,
            },
            "classifier": {
                "confidence_threshold": self.config.classifier_conf_threshold,
                "imgsz": self.config.classifier_imgsz,
            },
            "cooldown": {
                "frames": self.config.command_cooldown_frames,
                "last_command": self._last_command,
                "frames_since_command": self._frames_since_command,
            },
        }

    def process_frame(self, content: bytes) -> PipelineFrameResult:
        if self._last_command is not None:
            self._frames_since_command += 1

        request_id = self._new_request_id()
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        image = self._decode_image(content)
        frame_path = self._save_frame(image, request_id) if self.config.save_debug_artifacts else None

        self._ensure_models_loaded()
        prediction = self._spotter.predict(image)
        is_anomaly = item_present_from_prediction(prediction)

        self.window.append(
            FrameRecord(
                request_id=request_id,
                timestamp=timestamp,
                image=image.copy(),
                prediction=prediction,
                is_anomaly=is_anomaly,
                frame_path=frame_path,
            )
        )

        artifacts = {
            "frame_filename": None if frame_path is None else self._display_path(frame_path),
            "crop_filename": None,
            "debug_dir": None,
        }
        spotter_payload = self._spotter_payload(prediction)

        classifier_payload = self._empty_classifier_payload(ran=False)
        command = -1
        trigger_dir: Path | None = None

        if self.window.is_full and self.window.true_ratio > self.config.spotter_true_ratio_threshold:
            trigger_dir = self._make_trigger_dir(request_id) if self.config.save_debug_artifacts else None
            classifier_payload, crop_path = self._classify_window(trigger_dir)
            artifacts["crop_filename"] = None if crop_path is None else self._display_path(crop_path)
            artifacts["debug_dir"] = None if trigger_dir is None else self._display_path(trigger_dir)
            if classifier_payload["votes"]:
                counts = np.bincount(np.asarray(classifier_payload["votes"], dtype=int), minlength=3)
                command = int(np.argmax(counts))

        command, duplicate, suppressed_command = self._apply_cooldown(command)
        result = PipelineFrameResult(
            status="ok",
            command=command,
            label=command_to_label(command),
            spotter=spotter_payload,
            classifier=classifier_payload,
            artifacts=artifacts,
            duplicate=duplicate,
            suppressed_command=suppressed_command,
        )
        if trigger_dir is not None:
            (trigger_dir / "result.json").write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return result

    def _ensure_models_loaded(self) -> None:
        if self._spotter is None:
            if not self.config.spotter_config_path.exists():
                raise FileNotFoundError(f"Spotter config was not found: {self.config.spotter_config_path}")
            if not self.config.spotter_checkpoint_path.exists():
                raise FileNotFoundError(f"Spotter checkpoint was not found: {self.config.spotter_checkpoint_path}")

            from spotter import TorchSpotterPredictor, load_spotter_config

            spotter_config = load_spotter_config(
                self.config.spotter_config_path,
                workspace_root=self.config.project_root,
            )
            self._spotter = TorchSpotterPredictor(
                checkpoint_path=self.config.spotter_checkpoint_path,
                config=spotter_config,
                device=self.config.spotter_device,
            )

        if self._classifier is None:
            if not self.config.classifier_weights_path.exists():
                raise FileNotFoundError(f"Classifier weights were not found: {self.config.classifier_weights_path}")
            self._classifier = YoloMaterialClassifier(self.config.classifier_weights_path)

        if self.config.background_ref_path is not None and self._background is None:
            if not self.config.background_ref_path.exists():
                raise FileNotFoundError(f"Background reference was not found: {self.config.background_ref_path}")
            self._background = Image.open(self.config.background_ref_path).convert("RGB")
            self._background.load()

    def _classify_window(self, trigger_dir: Path | None) -> tuple[dict[str, Any], Path | None]:
        assert self._classifier is not None

        votes: list[int] = []
        confidences: list[float] = []
        details: list[dict[str, Any]] = []
        representative_crop_path: Path | None = None

        anomalous_records = [record for record in self.window.items if record.is_anomaly]
        for index, record in enumerate(anomalous_records):
            detail, crop_path = self._classify_record(
                record,
                record_index=index,
                trigger_dir=trigger_dir,
                save_root=index == 0,
            )
            details.append(detail)
            if representative_crop_path is None:
                representative_crop_path = crop_path

            classification = detail.get("classification")
            if not classification:
                continue
            confidence = float(classification["confidence"])
            command = int(classification["command"])
            if confidence < self.config.classifier_conf_threshold:
                detail["accepted"] = False
                detail["skip_reason"] = "low_confidence"
                continue

            detail["accepted"] = True
            votes.append(command)
            confidences.append(confidence)

        vote_counts_array = np.bincount(np.asarray(votes, dtype=int), minlength=3) if votes else np.zeros(3, dtype=int)
        return (
            {
                "ran": True,
                "votes": votes,
                "vote_counts": {
                    "other": int(vote_counts_array[0]),
                    "plastic": int(vote_counts_array[1]),
                    "paper": int(vote_counts_array[2]),
                },
                "confidence_mean": None if not confidences else float(np.mean(confidences)),
                "confidence_threshold": self.config.classifier_conf_threshold,
                "results": details,
            },
            representative_crop_path,
        )

    def _classify_record(
        self,
        record: FrameRecord,
        *,
        record_index: int,
        trigger_dir: Path | None,
        save_root: bool,
    ) -> tuple[dict[str, Any], Path | None]:
        assert self._classifier is not None

        crop_info, final_mask, heatmap = crop_from_spotter_maps(
            image=record.image,
            prediction=record.prediction,
            padding_ratio=self.config.crop_padding,
            min_padding=self.config.crop_min_padding,
        )
        crop_image = record.image.crop(crop_info.bbox)
        object_mask, background_mask, _, diff_threshold = build_clean_object_mask(
            image=record.image,
            background=self._background,
            crop_info=crop_info,
            spotter_mask=final_mask,
            clean_mode=self.config.clean_mode,
            threshold_min=self.config.background_threshold_min,
            threshold_max=self.config.background_threshold_max,
            anchor_dilate_ratio=self.config.anchor_dilate_ratio,
            align_background=self.config.align_background,
            use_edge_support=self.config.use_edge_support,
        )
        clean_crop = make_clean_crop(
            image=record.image,
            background=self._background,
            bbox=crop_info.bbox,
            object_mask=object_mask,
            fill_mode=self.config.background_fill,
        )

        crop_path: Path | None = None
        if trigger_dir is not None:
            record_dir = trigger_dir / "records" / f"{record_index:02d}_{record.request_id}"
            record_dir.mkdir(parents=True, exist_ok=True)
            crop_path = self._save_record_artifacts(
                output_dir=record_dir,
                record=record,
                crop_info=crop_info,
                crop_image=crop_image,
                clean_crop=clean_crop,
                final_mask=final_mask,
                object_mask=object_mask,
                background_mask=background_mask,
                heatmap=heatmap,
            )
            if save_root:
                self._save_record_artifacts(
                    output_dir=trigger_dir,
                    record=record,
                    crop_info=crop_info,
                    crop_image=crop_image,
                    clean_crop=clean_crop,
                    final_mask=final_mask,
                    object_mask=object_mask,
                    background_mask=background_mask,
                    heatmap=heatmap,
                )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "clean_crop.jpg"
                clean_crop.save(temp_path, quality=95)
                classification = self._classifier.classify_path(
                    temp_path,
                    imgsz=self.config.classifier_imgsz,
                    device=self.config.classifier_device,
                )
                return self._record_detail(record, crop_info, diff_threshold, classification.to_dict()), None

        classifier_input = crop_path or trigger_dir / "clean_crop.jpg"
        classification = self._classifier.classify_path(
            classifier_input,
            imgsz=self.config.classifier_imgsz,
            device=self.config.classifier_device,
        )
        detail = self._record_detail(record, crop_info, diff_threshold, classification.to_dict())
        if trigger_dir is not None:
            record_json_path = trigger_dir / "records" / f"{record_index:02d}_{record.request_id}" / "classification.json"
            record_json_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")
        return detail, crop_path

    def _save_record_artifacts(
        self,
        *,
        output_dir: Path,
        record: FrameRecord,
        crop_info: CropInfo,
        crop_image: Image.Image,
        clean_crop: Image.Image,
        final_mask: np.ndarray,
        object_mask: np.ndarray,
        background_mask: np.ndarray | None,
        heatmap: np.ndarray | None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        record.image.save(output_dir / "input.jpg", quality=95)
        crop_image.save(output_dir / "crop.jpg", quality=95)
        clean_crop.save(output_dir / "clean_crop.jpg", quality=95)
        Image.fromarray(final_mask, mode="L").save(output_dir / "spotter_mask.png")
        Image.fromarray(object_mask, mode="L").save(output_dir / "crop_mask.png")
        if background_mask is not None:
            Image.fromarray(background_mask, mode="L").save(output_dir / "background_mask.png")
        if heatmap is None:
            heatmap = resize_heatmap(getattr(record.prediction, "anomaly_map", None), record.image.size)
        if heatmap is not None:
            heatmap_image = Image.fromarray(heatmap_rgb(heatmap).astype(np.uint8), mode="RGB")
            heatmap_image.save(output_dir / "heatmap.jpg", quality=95)
        (output_dir / "crop.json").write_text(
            json.dumps(asdict(crop_info), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_dir / "clean_crop.jpg"

    def _record_detail(
        self,
        record: FrameRecord,
        crop_info: CropInfo,
        diff_threshold: float | None,
        classification: dict[str, Any],
    ) -> dict[str, Any]:
        crop_payload = asdict(crop_info)
        crop_payload["background_diff_threshold"] = diff_threshold
        return {
            "request_id": record.request_id,
            "timestamp": record.timestamp,
            "spotter": {
                "is_anomaly": record.is_anomaly,
                "score": get_prediction_score(record.prediction),
                "score_threshold": get_prediction_threshold(record.prediction, DEFAULT_SPOTTER_THRESHOLD),
            },
            "crop": crop_payload,
            "classification": classification,
        }

    def _apply_cooldown(self, command: int) -> tuple[int, bool, int | None]:
        if command < 0 or self.config.command_cooldown_frames <= 0:
            if command >= 0:
                self._last_command = command
                self._frames_since_command = 0
            return command, False, None

        if (
            self._last_command == command
            and self._frames_since_command <= self.config.command_cooldown_frames
        ):
            return -1, True, command

        self._last_command = command
        self._frames_since_command = 0
        return command, False, None

    def _spotter_payload(self, prediction: Any) -> dict[str, Any]:
        return {
            "window_size": self.window.maxlen,
            "filled": len(self.window),
            "true_count": self.window.true_count,
            "true_ratio": self.window.true_ratio,
            "threshold": self.config.spotter_true_ratio_threshold,
            "current_is_anomaly": item_present_from_prediction(prediction),
            "current_score": get_prediction_score(prediction),
            "current_score_threshold": get_prediction_threshold(prediction, DEFAULT_SPOTTER_THRESHOLD),
        }

    def _empty_classifier_payload(self, *, ran: bool) -> dict[str, Any]:
        return {
            "ran": ran,
            "votes": [],
            "vote_counts": {
                COMMAND_LABELS[0]: 0,
                COMMAND_LABELS[1]: 0,
                COMMAND_LABELS[2]: 0,
            },
            "confidence_mean": None,
            "confidence_threshold": self.config.classifier_conf_threshold,
        }

    def _save_frame(self, image: Image.Image, request_id: str) -> Path:
        date_part = datetime.now().strftime("%Y%m%d")
        frame_dir = self.config.debug_dir / "frames" / date_part
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"{request_id}.jpg"
        image.save(frame_path, quality=95)
        return frame_path

    def _make_trigger_dir(self, request_id: str) -> Path:
        trigger_dir = self.config.debug_dir / "triggers" / request_id
        trigger_dir.mkdir(parents=True, exist_ok=True)
        return trigger_dir

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.config.project_root).as_posix()
        except ValueError:
            return str(path.resolve())

    @staticmethod
    def _decode_image(content: bytes) -> Image.Image:
        if not content:
            raise ValueError("Frame body is empty.")
        try:
            with Image.open(BytesIO(content)) as image:
                image.load()
                return image.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Frame body is not a supported image.") from exc

    @staticmethod
    def _new_request_id() -> str:
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"


DEFAULT_SPOTTER_THRESHOLD = 0.5
