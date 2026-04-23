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

        uploads_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))
