from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.routers.ui import map_pipeline_result_to_ui_prediction, render_home_page
from app.services import ImageUploadError, image_events, image_storage
from app.services.pipeline_service import pipeline_service

router = APIRouter(tags=["images"])


async def store_image_and_broadcast(content: bytes, prediction: dict[str, str | int]) -> None:
    uploaded_image = image_storage.save_bytes(content)
    await image_events.broadcast_image_uploaded(uploaded_image, prediction)


async def read_upload_or_raw_body(
    request: Request,
    image: UploadFile | None,
) -> bytes:
    if image is not None:
        content = await image.read()
        await image.close()
        return content
    return await request.body()


async def infer_and_store(content: bytes) -> dict[str, str | int]:
    pipeline_result = await run_in_threadpool(pipeline_service.process_frame, content)
    prediction = map_pipeline_result_to_ui_prediction(pipeline_result.command, pipeline_result.label)
    await store_image_and_broadcast(content, prediction)
    return prediction


@router.post("/images/upload")
async def upload_image(
    request: Request,
    image: Annotated[UploadFile, File()],
):
    try:
        content = await read_upload_or_raw_body(request, image)
        prediction = await infer_and_store(content)
    except ImageUploadError as exc:
        return render_home_page(request, error_message=str(exc), status_code=400)
    except ValueError as exc:
        return render_home_page(request, error_message=str(exc), status_code=422)
    except FileNotFoundError as exc:
        return render_home_page(request, error_message=str(exc), status_code=503)
    except Exception as exc:
        return render_home_page(request, error_message=f"Inference failed: {exc}", status_code=500)

    return render_home_page(request, prediction=prediction)


@router.post("/api/images/upload")
async def api_upload_image(
    request: Request,
    image: Annotated[UploadFile | None, File()] = None,
):
    try:
        content = await read_upload_or_raw_body(request, image)
        prediction = await infer_and_store(content)
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
            "class_id": prediction["class_id"],
            "label": prediction["label"],
        },
    )


@router.post("/api/images/upload/raw")
async def api_upload_image_raw(request: Request):
    try:
        content = await read_upload_or_raw_body(request, image=None)
        prediction = await infer_and_store(content)
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
            "class_id": prediction["class_id"],
            "label": prediction["label"],
        },
    )
