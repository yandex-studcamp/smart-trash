import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services import image_events

router = APIRouter()


@router.websocket("/ws/images")
async def image_updates(websocket: WebSocket):
    await image_events.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        await image_events.disconnect(websocket)
    except asyncio.CancelledError:
        await image_events.disconnect(websocket)
        raise
