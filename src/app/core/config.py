import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates


load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return int(raw_value)


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return float(raw_value)


def _get_optional_str(name: str, default: str | None = None) -> str | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    value = raw_value.strip()
    return value or default


def _get_list(name: str, default: list[str]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return tuple(default)

    parts = [part.strip() for part in raw_value.split(",")]
    return tuple(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "Smart Trash Server"))
    app_description: str = field(
        default_factory=lambda: os.getenv(
            "APP_DESCRIPTION",
            "Сервер для загрузки, просмотра и дальнейшей обработки изображений.",
        )
    )
    app_version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))
    host: str = field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _get_int("APP_PORT", 8000))
    reload: bool = field(default_factory=lambda: _get_bool("APP_RELOAD", False))
    cors_allow_origins: tuple[str, ...] = field(default_factory=lambda: _get_list("APP_CORS_ALLOW_ORIGINS", ["*"]))
    max_upload_size_bytes: int = field(default_factory=lambda: _get_int("APP_MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024))
    app_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    templates_dir: Path = field(init=False)
    static_dir: Path = field(init=False)
    uploads_dir: Path = field(init=False)
    pipeline_spotter_window_size: int = field(default_factory=lambda: _get_int("PIPELINE_SPOTTER_WINDOW_SIZE", 8))
    pipeline_spotter_true_ratio_threshold: float = field(
        default_factory=lambda: _get_float("PIPELINE_SPOTTER_TRUE_RATIO_THRESHOLD", 0.6)
    )
    pipeline_spotter_config: Path = field(
        default_factory=lambda: Path(os.getenv("PIPELINE_SPOTTER_CONFIG", "src/config/spotter_patchcore.yaml"))
    )
    pipeline_spotter_checkpoint: Path = field(
        default_factory=lambda: Path(os.getenv("PIPELINE_SPOTTER_CHECKPOINT", "src/spotter/meta/patchcore.ckpt"))
    )
    pipeline_spotter_device: str = field(default_factory=lambda: os.getenv("PIPELINE_SPOTTER_DEVICE", "auto"))
    pipeline_classifier_weights: Path = field(
        default_factory=lambda: Path(os.getenv("PIPELINE_CLASSIFIER_WEIGHTS", "src/classifier/weights/best.pt"))
    )
    pipeline_classifier_device: str | None = field(
        default_factory=lambda: _get_optional_str("PIPELINE_CLASSIFIER_DEVICE")
    )
    pipeline_classifier_imgsz: int = field(default_factory=lambda: _get_int("PIPELINE_CLASSIFIER_IMGSZ", 224))
    pipeline_classifier_conf_threshold: float = field(
        default_factory=lambda: _get_float("PIPELINE_CLASSIFIER_CONF_THRESHOLD", 0.3)
    )
    pipeline_crop_padding: float = field(default_factory=lambda: _get_float("PIPELINE_CROP_PADDING", 0.10))
    pipeline_crop_min_padding: int = field(default_factory=lambda: _get_int("PIPELINE_CROP_MIN_PADDING", 6))
    pipeline_save_debug_artifacts: bool = field(
        default_factory=lambda: _get_bool("PIPELINE_SAVE_DEBUG_ARTIFACTS", True)
    )
    pipeline_debug_dir: Path = field(
        default_factory=lambda: Path(os.getenv("PIPELINE_DEBUG_DIR", "runs/runtime_pipeline"))
    )
    pipeline_background_fill: str = field(default_factory=lambda: os.getenv("PIPELINE_BACKGROUND_FILL", "black"))
    pipeline_clean_mode: str = field(default_factory=lambda: os.getenv("PIPELINE_CLEAN_MODE", "strict"))
    pipeline_background_ref: Path | None = field(
        default_factory=lambda: None
        if _get_optional_str("PIPELINE_BACKGROUND_REF") is None
        else Path(_get_optional_str("PIPELINE_BACKGROUND_REF") or "")
    )
    pipeline_command_cooldown_frames: int = field(
        default_factory=lambda: _get_int("PIPELINE_COMMAND_COOLDOWN_FRAMES", 0)
    )

    def __post_init__(self) -> None:
        templates_dir = self.app_dir / "templates"
        static_dir = self.app_dir / "static"
        uploads_dir_value = os.getenv("APP_UPLOADS_DIR")
        uploads_dir = Path(uploads_dir_value) if uploads_dir_value else self.project_root / "data" / "uploads"
        if not uploads_dir.is_absolute():
            uploads_dir = self.project_root / uploads_dir

        object.__setattr__(self, "templates_dir", templates_dir)
        object.__setattr__(self, "static_dir", static_dir)
        object.__setattr__(self, "uploads_dir", uploads_dir.resolve())

        for attr_name in (
            "pipeline_spotter_config",
            "pipeline_spotter_checkpoint",
            "pipeline_classifier_weights",
            "pipeline_debug_dir",
        ):
            path = getattr(self, attr_name)
            if not path.is_absolute():
                path = self.project_root / path
            object.__setattr__(self, attr_name, path.resolve())

        if self.pipeline_background_ref is not None:
            background_ref = self.pipeline_background_ref
            if not background_ref.is_absolute():
                background_ref = self.project_root / background_ref
            object.__setattr__(self, "pipeline_background_ref", background_ref.resolve())

        uploads_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_debug_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))
