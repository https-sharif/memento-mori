from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google.genai import types

import store
from gemini_client import GEMINI_MODEL, client
from models import AskRequest, BehavioralAnalysis, CaregiverInput, MemoryCard
from patient_memory import PatientMemoryStore
from perception import PerceptionStateTracker

load_dotenv()

logger = logging.getLogger(__name__)

patient_memory = PatientMemoryStore()

# Bounded so a long-running process can't grow the cache without limit —
# the key is (scene signature x question), which is effectively unbounded.
_ASK_CACHE_MAX = 256
_ask_cache_lock = Lock()
_ask_cache: OrderedDict[str, dict] = OrderedDict()


def _cache_get(key: str) -> dict | None:
    with _ask_cache_lock:
        if key not in _ask_cache:
            return None
        _ask_cache.move_to_end(key)
        return _ask_cache[key]


def _cache_put(key: str, value: dict) -> None:
    with _ask_cache_lock:
        _ask_cache[key] = value
        _ask_cache.move_to_end(key)
        while len(_ask_cache) > _ASK_CACHE_MAX:
            _ask_cache.popitem(last=False)


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def _scene_signature(scene: dict) -> str:
    if not isinstance(scene, dict):
        return "{}"

    person = scene.get("person", {}) if isinstance(scene.get("person", {}), dict) else {}
    objects = scene.get("objects", []) if isinstance(scene.get("objects", []), list) else []
    normalized_objects = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        normalized_objects.append({
            "label": item.get("label", ""),
            "confidence": round(float(item.get("confidence", 0.0)), 3) if item.get("confidence") is not None else None,
        })
    payload = {
        "recognized": bool(person.get("recognized")),
        "face_detected": bool(person.get("face_detected")),
        "name": person.get("name", ""),
        "relationship": person.get("relationship", ""),
        "note": person.get("note", ""),
        "objects": normalized_objects,
    }
    return json.dumps(payload, sort_keys=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracker = PerceptionStateTracker(memory=patient_memory)
    task = asyncio.create_task(tracker.listen_and_process())
    yield
    task.cancel()
    # Await the cancellation so shutdown doesn't race the perception loop.
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="Dementia Memory Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/latest")
async def latest(kind: str = Query("patient", alias="type")):
    payload = store.get_latest_caretaker_advice() if kind == "caretaker" else store.get_latest_patient_card()
    if not payload:
        return {"status": "empty"}
    return payload


@app.post("/ask", response_model=MemoryCard)
async def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    scene = store.get_latest_scene()
    question_key = _normalize_question(question)
    cache_key = f"{_scene_signature(scene)}::{question_key}"

    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    person = scene.get("person", {}) if isinstance(scene, dict) else {}
    objects = scene.get("objects", []) if isinstance(scene, dict) else []
    retrieval_query = f"{question}\nPerson: {person}\nObjects: {objects}"
    retrieved_context = patient_memory.retrieve_text(retrieval_query, top_k=5)

    prompt = (
        "You are a calm dementia-care assistant. Give practical, non-judgmental guidance for the caretaker. "
        "Do not mention that you are an AI or reference hidden chain of thought.\n"
        f"Retrieved patient memory context:\n{retrieved_context}\n"
        f"Current scene: {json.dumps(scene)}\n"
        f"Caretaker question: {question}"
    )

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You answer caretakers with short, concrete suggestions. "
                    "Focus on safety, reassurance, and immediate next steps. "
                    "Keep the response short and speak directly to the caretaker."
                ),
                response_mime_type="application/json",
                response_schema=MemoryCard,
                temperature=0.2,
            ),
        )
    except Exception as exc:
        logger.exception("Gemini call failed for /ask")
        raise HTTPException(status_code=503, detail=f"Assistant service unavailable: {exc}") from exc

    # .parsed is None when the model is safety-blocked or returns unparseable JSON.
    if response.parsed is None:
        logger.warning("Gemini returned no parseable content for /ask (question=%r)", question)
        raise HTTPException(
            status_code=503,
            detail="The assistant could not produce an answer for that question. Please rephrase and try again.",
        )

    result = response.parsed.model_dump()
    store.set_latest_caretaker_advice(result)
    _cache_put(cache_key, result)
    return result


@app.post("/api/caregiver/analyze", response_model=BehavioralAnalysis)
async def analyze_caregiver_input(payload: CaregiverInput):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Context: {patient_memory.profile_text()}\nInput: {message}",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a clinical expert system in dementia caregiving. Analyze the user's input. "
                    "Prioritize de-escalation, behavioral redirection, and validating caregiver stress. "
                    "Never suggest medical prescriptions or changing drug dosages."
                ),
                response_mime_type="application/json",
                response_schema=BehavioralAnalysis,
                temperature=0.1,
            ),
        )
    except Exception as exc:
        logger.exception("Gemini call failed for /api/caregiver/analyze")
        raise HTTPException(status_code=503, detail=f"Assistant service unavailable: {exc}") from exc

    if response.parsed is None:
        logger.warning("Gemini returned no parseable content for /api/caregiver/analyze")
        raise HTTPException(
            status_code=503,
            detail="The assistant could not analyze that report. Please rephrase and try again.",
        )

    return response.parsed
