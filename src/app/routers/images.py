from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.routers.ui import render_home_page
from app.services import ImageUploadError, image_events, image_storage
from app.services.pipeline_service import pipeline_service

router = APIRouter(tags=["images"])


async def store_image_and_broadcast(content: bytes) -> None:
    uploaded_image = image_storage.save_bytes(content)
    await image_events.broadcast_image_uploaded(uploaded_image)


async def read_upload_or_raw_body(
    request: Request,
    image: UploadFile | None,
) -> bytes:
    if image is not None:
        content = await image.read()
        await image.close()
        return content
    return await request.body()


def map_pipeline_result_to_remote_class(command: int, label: str | None) -> tuple[int, str]:
    normalized_label = (label or "").strip().lower()
    if command < 0 or normalized_label in {"none", "nothing"}:
        return 3, "nothing"
    if normalized_label == "paper":
        return 1, "paper"
    if normalized_label == "plastic":
        return 2, "plastic"
    if normalized_label == "other":
        return 0, "other"

    if command == 2:
        return 1, "paper"
    if command == 1:
        return 2, "plastic"
    if command == 0:
        return 0, "other"
    return 3, "nothing"


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
    image: Annotated[UploadFile | None, File()] = None,
):
    try:
        content = await read_upload_or_raw_body(request, image)
        await store_image_and_broadcast(content)
        pipeline_result = await run_in_threadpool(pipeline_service.process_frame, content)
        class_id, class_label = map_pipeline_result_to_remote_class(
            pipeline_result.command,
            pipeline_result.label,
        )
    except ImageUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "class_id": class_id,
            "label": class_label,
        },
    )


@router.post("/api/images/upload/raw")
async def api_upload_image_raw(request: Request):
    try:
        content = await read_upload_or_raw_body(request, image=None)
        await store_image_and_broadcast(content)
        pipeline_result = await run_in_threadpool(pipeline_service.process_frame, content)
        class_id, class_label = map_pipeline_result_to_remote_class(
            pipeline_result.command,
            pipeline_result.label,
        )
    except ImageUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "class_id": class_id,
            "label": class_label,
        },
    )
