from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routers.images import router as images_router
from app.routers.pages import router as pages_router
from app.routers.pipeline import router as pipeline_router
from app.routers.realtime import router as realtime_router


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(pages_router)
    application.include_router(images_router)
    application.include_router(pipeline_router)
    application.include_router(realtime_router)

    application.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    application.mount("/uploads", StaticFiles(directory=str(settings.uploads_dir)), name="uploads")

    return application


app = create_app()
