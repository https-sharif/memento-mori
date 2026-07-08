"""LLM service entrypoint.

Run this after starting vision_service:

    export GEMINI_API_KEY=...
    export VISION_WS_URL=ws://localhost:8000/ws
    python main.py
"""

import asyncio
import os
from threading import Thread

from http_server import start_http_server
from perception import PerceptionStateTracker


def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY before running.")
        raise SystemExit(1)

    http_host = os.environ.get("LLM_HTTP_HOST", "127.0.0.1")
    http_port = int(os.environ.get("LLM_HTTP_PORT", "8001"))

    server = start_http_server(http_host, http_port)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"LLM latest-output server listening on http://{http_host}:{http_port}")

    tracker = PerceptionStateTracker()

    try:
        asyncio.run(tracker.listen_and_process())
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()