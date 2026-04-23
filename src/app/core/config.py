from dataclasses import dataclass, field
from pathlib import Path

from fastapi.templating import Jinja2Templates


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "Smart Trash Server"
    app_description: str = "Сервер для загрузки, просмотра и дальнейшей обработки изображений."
    app_version: str = "1.0.0"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    app_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    templates_dir: Path = field(init=False)
    static_dir: Path = field(init=False)
    uploads_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        templates_dir = self.app_dir / "templates"
        static_dir = self.app_dir / "static"
        uploads_dir = self.project_root / "data" / "uploads"

        object.__setattr__(self, "templates_dir", templates_dir)
        object.__setattr__(self, "static_dir", static_dir)
        object.__setattr__(self, "uploads_dir", uploads_dir)

        uploads_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))
