from fastapi import Request

from app.core.config import templates
from app.services.image_service import StoredImage, image_storage


def render_home_page(
    request: Request,
    *,
    uploaded_image: StoredImage | None = None,
    error_message: str | None = None,
    status_code: int = 200,
):
    current_image = uploaded_image or image_storage.get_latest_image()

    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "uploaded_image": current_image,
            "error_message": error_message,
        },
        status_code=status_code,
    )
