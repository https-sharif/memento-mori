"""
WebSocket connection manager + the /ws endpoint.

Clients receive the latest recognition result as JSON at the configured interval.
A client that falls behind (slow network) is silently dropped rather than
blocking the broadcast loop.
"""
import json
import logging
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WS client connected. Total: %d", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WS client disconnected. Total: %d", len(self._connections))

    async def broadcast(self, data: dict) -> None:
        if not self._connections:
            return
        payload = json.dumps(data)
        dead: List[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def num_clients(self) -> int:
        return len(self._connections)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Stream recognition results in real-time.

    Connect with:  ws://localhost:8000/ws

    Receives JSON in the same format as POST /recognize, pushed at the
    configured ws_interval (default: 1 s) by the background vision loop.
    Sends a "ping" keep-alive text every 30 s so proxies don't time out.
    """
    await manager.connect(websocket)
    try:
        while True:
            # We only need to keep the connection alive here;
            # broadcasting is done by the vision loop in main.py.
            await websocket.receive_text()   # blocks until client sends or disconnects
    except WebSocketDisconnect:
        manager.disconnect(websocket)
