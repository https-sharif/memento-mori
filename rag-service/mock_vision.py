import asyncio
import json
import time
from websockets.asyncio.server import serve


async def generate_frames(websocket):
    print("Connection established with Perception Engine.")
    scenarios = [
        {"person": {"recognized": False, "name": "Unknown", "relationship": "", "note": ""}, "objects": []},
        {"person": {"recognized": False, "name": "Unknown", "relationship": "", "note": ""}, "objects": []},
        {"person": {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "Visits on weekends"}, "objects": []},
        {"person": {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "Visits on weekends"}, "objects": []},
        {"person": {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "Visits on weekends"}, "objects": [{"label": "Medicine Bottle"}]},
        {"person": {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "Visits on weekends"}, "objects": [{"label": "Medicine Bottle"}]},
        {"person": {"recognized": False, "name": "Unknown", "relationship": "", "note": ""}, "objects": []},
    ]

    while True:
        for frame in scenarios:
            frame["timestamp"] = int(time.time())
            await websocket.send(json.dumps(frame))
            print("Frame dispatched from camera feed...")
            await asyncio.sleep(2)


async def main():
    async with serve(generate_frames, "localhost", 8000):
        print("Local Mock Vision Server running on ws://localhost:8000/ws")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping Mock Vision Server.")
