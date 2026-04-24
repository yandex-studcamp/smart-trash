from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


COMMAND_LABELS = {
    -1: "none",
    0: "other",
    1: "plastic",
    2: "paper",
}


@dataclass(slots=True)
class ClassificationResult:
    raw_index: int
    raw_label: str
    command: int
    label: str
    confidence: float
    top5: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def command_to_label(command: int) -> str:
    return COMMAND_LABELS.get(int(command), "other")


def normalize_label_to_command(label: str) -> int:
    normalized = str(label).strip().lower()
    if "plastic" in normalized:
        return 1
    if "paper" in normalized or "cardboard" in normalized:
        return 2
    if "other" in normalized:
        return 0
    return 0


def _model_names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    return {}


def _tensor_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


class YoloMaterialClassifier:
    def __init__(self, weights_path: str | Path) -> None:
        from ultralytics import YOLO

        self.weights_path = Path(weights_path).resolve()
        self.model = YOLO(str(self.weights_path))

    @property
    def names(self) -> dict[int, str]:
        return _model_names_to_dict(getattr(self.model, "names", {}))

    def classify_path(
        self,
        image_path: str | Path,
        *,
        imgsz: int,
        device: str | None,
    ) -> ClassificationResult:
        predict_kwargs: dict[str, Any] = {
            "source": str(image_path),
            "imgsz": int(imgsz),
            "verbose": False,
        }
        if device:
            predict_kwargs["device"] = device

        result = self.model.predict(**predict_kwargs)[0]
        probs = result.probs
        raw_index = int(probs.top1)
        confidence = float(probs.top1conf)
        names = self.names
        raw_label = names.get(raw_index, str(raw_index))
        command = normalize_label_to_command(raw_label)

        top5: list[dict[str, Any]] = []
        top5_indices = [int(index) for index in probs.top5]
        top5_confidences = _tensor_values(probs.top5conf)
        for index, top_confidence in zip(top5_indices, top5_confidences):
            top5.append(
                {
                    "index": index,
                    "label": names.get(index, str(index)),
                    "confidence": float(top_confidence),
                    "command": normalize_label_to_command(names.get(index, str(index))),
                }
            )

        return ClassificationResult(
            raw_index=raw_index,
            raw_label=raw_label,
            command=command,
            label=command_to_label(command),
            confidence=confidence,
            top5=top5,
        )
