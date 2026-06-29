# Dementia Assistant

A web-based AI assistant that helps people with dementia recognize familiar faces and objects, answer simple questions by voice, and follow daily routines.

---

## Who owns what

| Part | Owner | Status |
|------|-------|--------|
| Vision Service (face + object recognition) | Sharif | Done |
| LLM / RAG / NLP | — | Todo |
| Frontend | — | Todo |

---

## How it all connects

```
Webcam / Photo
      │
      ▼
Vision Service  →  JSON { person, objects }
      │
      ├──► LLM / RAG  →  memory card text, voice answer
      │
      └──► Frontend   →  shows card, plays audio
```

The Vision Service is a standalone HTTP server. Everyone else just calls it — no ML setup needed on your end.

---

## Running the vision service

**Mac**
```bash
cd vision_service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Windows**
```bat
cd vision_service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Service runs at **http://localhost:8000** on both platforms.

---

## What the vision service gives you

Every call returns this:

```json
{
  "person": {
    "recognized": true,
    "name": "Sarah",
    "relationship": "Daughter",
    "note": "Visits on weekends",
    "confidence": 0.96,
    "face_detected": true
  },
  "objects": [
    { "label": "Medicine Bottle", "confidence": 0.91 }
  ],
  "timestamp": 1751190195
}
```

---

## For the LLM / RAG person

**Get the current scene:**
```python
import httpx
result = httpx.post("http://localhost:8000/recognize").json()
person = result["person"]
objects = result["objects"]
```

**Build context for your LLM:**
```python
context = ""

if person["recognized"]:
    context += f"The person is {person['name']}, {person['relationship']}."
    if person["note"]:
        context += f" Note: {person['note']}."
else:
    context += "The person is not recognized."

if objects:
    context += " Visible: " + ", ".join(o["label"] for o in objects) + "."

# use `context` in your system prompt or RAG retrieval
```

**Or subscribe to live updates:**
```python
import asyncio, json, websockets

async def listen():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        async for msg in ws:
            result = json.loads(msg)  # fires every ~1 second
```

Use `person["name"]` as the key to retrieve caregiver notes from your vector store.

---

## For the frontend person

Connect to the WebSocket on page load — results push every ~1 second:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = ({ data }) => {
  const { person, objects } = JSON.parse(data);

  if (person.recognized) {
    showMemoryCard(person.name, person.relationship, person.note);
  } else if (person.face_detected) {
    showMemoryCard("Unknown person");
  } else {
    clearCard();
  }

  showObjects(objects.map(o => o.label));
};
```

For a manual snapshot button:
```javascript
const res = await fetch("http://localhost:8000/recognize", { method: "POST" });
const data = await res.json();
```

CORS is open — any origin works.

---

## Registering a face

Sharif handles this. Open `http://localhost:8000/static/register.html` and fill in name, relationship, and capture a few photos.

To see who's registered:
```bash
curl http://localhost:8000/people
```
