from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.routers.ui import render_home_page
from app.services import ImageUploadError, image_events, image_storage
from app.services.image_service import StoredImage

router = APIRouter(tags=["images"])


def build_image_payload(request: Request, image: StoredImage) -> dict[str, str]:
    absolute_url = str(request.url_for("uploads", path=image.filename))
    return {
        "filename": image.filename,
        "url": image.url,
        "absolute_url": absolute_url,
    }


async def store_image_and_broadcast(content: bytes) -> StoredImage:
    uploaded_image = image_storage.save_bytes(content)
    await image_events.broadcast_image_uploaded(uploaded_image)
    return uploaded_image


@router.post("/images/upload")
async def upload_image(
    request: Request,
    image: Annotated[UploadFile, File()],
):
    try:
        uploaded_image = await image_storage.save_upload(image)
        await image_events.broadcast_image_uploaded(uploaded_image)
    except ImageUploadError as exc:
        return render_home_page(request, error_message=str(exc), status_code=400)

    return render_home_page(request, uploaded_image=uploaded_image)


@router.post("/api/images/upload")
async def api_upload_image(
    request: Request,
    image: Annotated[UploadFile, File()],
):
    try:
        uploaded_image = await image_storage.save_upload(image)
        await image_events.broadcast_image_uploaded(uploaded_image)
    except ImageUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        status_code=201,
        content={
            "status": "ok",
            "image": build_image_payload(request, uploaded_image),
        },
    )


@router.post("/api/images/upload/raw")
async def api_upload_image_raw(request: Request):
    try:
        uploaded_image = await store_image_and_broadcast(await request.body())
    except ImageUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        status_code=201,
        content={
            "status": "ok",
            "image": build_image_payload(request, uploaded_image),
        },
    )
