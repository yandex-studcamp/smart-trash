from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile

from app.routers.ui import render_home_page
from app.services.image_service import ImageUploadError, image_storage

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload")
async def upload_image(
    request: Request,
    image: Annotated[UploadFile, File()],
):
    try:
        uploaded_image = await image_storage.save_upload(image)
    except ImageUploadError as exc:
        return render_home_page(request, error_message=str(exc), status_code=400)

    return render_home_page(request, uploaded_image=uploaded_image)
