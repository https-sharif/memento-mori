import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

from app.frontend_hub import FrontendHub
from app.models import BehavioralAnalysis, CaregiverInput, MemoryCardOutput, RAGChatRequest, RAGChatResponse
from app.rag_chatbot import RAGMemoryChatbot

load_dotenv()

app = FastAPI(title="Dementia Memory Assistant")
hub = FrontendHub()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"ok": True}


@app.websocket("/ws/frontend")
async def frontend_socket(websocket: WebSocket):
    await hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(websocket)


@app.post("/api/cue")
async def receive_memory_cue(cue: MemoryCardOutput):
    payload = cue.model_dump()
    await hub.broadcast(payload)
    return {"ok": True, "broadcast": payload}


@app.post("/api/rag/orientation", response_model=RAGChatResponse)
async def rag_orientation_chat(payload: RAGChatRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        fallback = MemoryCardOutput(
            card_title="Configuration needed",
            card_body="The RAG chatbot needs a Gemini API key before it can generate cards.",
            voice_guidance="The assistant is not connected yet. Please ask the caregiver to check the setup.",
        )
        return RAGChatResponse(
            output=fallback,
            debug={
                "retrieved_context": "GEMINI_API_KEY is missing, so retrieval/LLM generation did not run.",
                "llm_input": payload.message,
            },
        )

    chatbot = RAGMemoryChatbot()
    result = chatbot.generate_structured_output(payload)
    await hub.broadcast(result.output.model_dump())
    return result


@app.post("/api/caregiver/analyze", response_model=BehavioralAnalysis)
async def analyze_caregiver_input(payload: CaregiverInput):
    if not os.environ.get("GEMINI_API_KEY"):
        return BehavioralAnalysis(
            category="Configuration Error",
            observed_triggers=["Missing GEMINI_API_KEY"],
            clinical_rationale="The caregiver analysis model cannot run until the API key is configured.",
            actionable_intervention="Set GEMINI_API_KEY in your environment or .env file, then restart the server.",
            is_crisis=False,
        )

    client = genai.Client()
    patient_profile = (
        "Patient: Arthur. Diagnosed with Alzheimer's. Needs calm, short, non-clinical reassurance."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Context: {patient_profile}\nInput: {payload.message}",
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
    return response.parsed
