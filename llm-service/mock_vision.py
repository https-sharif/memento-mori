import asyncio
import json
import time
from websockets.asyncio.server import serve

async def generate_frames(websocket):
    print("Connection established with Perception Engine.")
    
    # 1. Timeline of simulated visual telemetry frames
    scenarios = [
        # Frame 1 & 2: Baseline empty room
        {"person": {"recognized": False, "name": "Unknown", "relationship": "", "note": ""}, "objects": []},
        {"person": {"recognized": False, "name": "Unknown", "relationship": "", "note": ""}, "objects": []},
        
        # Frame 3 & 4: Sarah walks in
        {"person": {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "Visits on weekends"}, "objects": []},
        {"person": {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "Visits on weekends"}, "objects": []},
        
        # Frame 5 & 6: Sarah hands over a medicine bottle
        {"person": {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "Visits on weekends"}, "objects": [{"label": "Medicine Bottle"}]},
        {"person": {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "Visits on weekends"}, "objects": [{"label": "Medicine Bottle"}]},
        
        # Frame 7: Room empties out again
        {"person": {"recognized": False, "name": "Unknown", "relationship": "", "note": ""}, "objects": []}
    ]

    for frame in scenarios:
        frame["timestamp"] = int(time.time())
        await websocket.send(json.dumps(frame))
        print("Frame dispatched from camera feed...")
        await asyncio.sleep(2) # Emit telemetry at standard intervals

async def main():
    async with serve(generate_frames, "localhost", 8000) as server:
        print("Local Mock Vision Server running on ws://localhost:8000")
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping Mock Vision Server.")