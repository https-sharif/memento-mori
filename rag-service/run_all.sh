#!/usr/bin/env bash
set -euo pipefail

python mock_vision.py &
VISION_PID=$!
uvicorn app.main:app --host 0.0.0.0 --port 8080 &
API_PID=$!
sleep 2
python perception_service.py

trap 'kill $VISION_PID $API_PID 2>/dev/null || true' EXIT
