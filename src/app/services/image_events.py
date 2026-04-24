from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket

from app.services.image_service import StoredImage


class ImageEventBroadcaster:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast_image_uploaded(self, image: StoredImage, prediction: dict[str, Any] | None = None) -> None:
        payload = {
            "type": "image_uploaded",
            "image": {
                "filename": image.filename,
                "url": image.url,
            },
        }
        if prediction is not None:
            payload["prediction"] = prediction

        async with self._lock:
            connections = list(self._connections)

        stale_connections: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale_connections.append(websocket)

        if not stale_connections:
            return

        async with self._lock:
            for websocket in stale_connections:
                self._connections.discard(websocket)


image_events = ImageEventBroadcaster()
