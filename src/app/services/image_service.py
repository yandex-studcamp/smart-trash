from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import settings


class ImageUploadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StoredImage:
    filename: str
    url: str


class ImageStorageService:
    _format_to_suffix = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
    }

    def __init__(self, upload_dir: Path, base_url: str, max_upload_size_bytes: int) -> None:
        self.upload_dir = upload_dir
        self.base_url = base_url.rstrip("/")
        self.max_upload_size_bytes = max_upload_size_bytes

    async def save_upload(self, upload: UploadFile) -> StoredImage:
        content = await upload.read()
        await upload.close()
        return self.save_bytes(content)

    def save_bytes(self, content: bytes) -> StoredImage:
        if not content:
            raise ImageUploadError("Файл пустой. Выберите изображение и попробуйте ещё раз.")

        if len(content) > self.max_upload_size_bytes:
            max_size_mb = self.max_upload_size_bytes // (1024 * 1024)
            raise ImageUploadError(f"Файл слишком большой. Максимальный размер: {max_size_mb} MB.")

        image_format = self._inspect_image(content)
        filename = f"{uuid4().hex}{self._format_to_suffix[image_format]}"
        file_path = self.upload_dir / filename
        file_path.write_bytes(content)

        return StoredImage(
            filename=filename,
            url=f"{self.base_url}/{filename}",
        )

    def get_latest_image(self) -> StoredImage | None:
        image_paths = sorted(
            (
                path
                for path in self.upload_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self._format_to_suffix.values()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for path in image_paths:
            try:
                return self._build_stored_image(path)
            except ImageUploadError:
                continue

        return None

    def _inspect_image(self, content: bytes) -> str:
        try:
            with Image.open(BytesIO(content)) as image:
                image.load()
                image_format = (image.format or "").upper()
                width, height = image.size
        except UnidentifiedImageError as exc:
            raise ImageUploadError(
                "Файл не распознан как изображение. Поддерживаются JPG, PNG, WEBP и GIF."
            ) from exc

        if image_format not in self._format_to_suffix:
            raise ImageUploadError("Поддерживаются только JPG, PNG, WEBP и GIF.")

        if width <= 0 or height <= 0:
            raise ImageUploadError("Не удалось определить размеры изображения.")

        return image_format

    def _build_stored_image(self, file_path: Path) -> StoredImage:
        try:
            with Image.open(file_path) as image:
                image.load()
                image_format = (image.format or file_path.suffix.lstrip(".")).upper()
        except (OSError, UnidentifiedImageError) as exc:
            raise ImageUploadError(f"Не удалось прочитать изображение {file_path.name}.") from exc

        if image_format not in self._format_to_suffix:
            raise ImageUploadError(f"Неподдерживаемый формат файла {file_path.name}.")

        return StoredImage(
            filename=file_path.name,
            url=f"{self.base_url}/{file_path.name}",
        )


image_storage = ImageStorageService(
    upload_dir=settings.uploads_dir,
    base_url="/uploads",
    max_upload_size_bytes=settings.max_upload_size_bytes,
)
