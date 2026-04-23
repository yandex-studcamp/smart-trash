from fastapi import APIRouter, Request

from app.routers.ui import render_home_page

router = APIRouter()


@router.get("/")
async def home(request: Request):
    return render_home_page(request)
