import json
from fastapi import WebSocket


class FrontendHub:
    def __init__(self):
        self.connections: set[WebSocket] = set()
        self.latest_payload: dict | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)
        if self.latest_payload:
            await websocket.send_json(self.latest_payload)

    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)

    async def broadcast(self, payload: dict):
        self.latest_payload = payload
        stale: list[WebSocket] = []
        for ws in self.connections:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)
