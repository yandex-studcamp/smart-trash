from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.services.pipeline_service import pipeline_service

router = APIRouter(tags=["pipeline"])


@router.post("/api/pipeline/frame")
async def pipeline_frame(
    request: Request,
    image: Annotated[UploadFile | None, File()] = None,
):
    try:
        if image is not None:
            content = await image.read()
            await image.close()
        else:
            content = await request.body()

        result = await run_in_threadpool(pipeline_service.process_frame, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    return JSONResponse(content=result.to_dict())


@router.get("/api/pipeline/state")
async def pipeline_state():
    return await run_in_threadpool(pipeline_service.state)


@router.post("/api/pipeline/reset")
async def pipeline_reset():
    return await run_in_threadpool(pipeline_service.reset)
