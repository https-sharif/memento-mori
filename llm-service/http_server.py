from __future__ import annotations

import json
from urllib.parse import parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from google import genai
from google.genai import types

from assistant import CaretakerAdvice
from store import get_latest_output, get_latest_scene, set_latest_output


_client = genai.Client()
_cache_lock = Lock()
_ask_cache: dict[str, dict] = {}


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


def _build_caretaker_prompt(question: str, scene: dict) -> str:
    person = scene.get("person", {}) if isinstance(scene, dict) else {}
    objects = scene.get("objects", []) if isinstance(scene, dict) else []
    scene_summary = {
        "person": person,
        "objects": objects,
        "timestamp": scene.get("timestamp") if isinstance(scene, dict) else None,
    }
    return (
        "You are a calm dementia-care assistant. Give practical, non-judgmental guidance for the caretaker. "
        "Do not mention that you are an AI or reference hidden chain of thought. "
        f"Current scene: {json.dumps(scene_summary)}\n"
        f"Caretaker question: {question}"
    )


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/health"}:
            self._send_json(200, {"status": "ok"})
            return

        if self.path == "/latest":
            payload = get_latest_output()
            if not payload:
                self._send_json(200, {"status": "empty"})
                return
            self._send_json(200, payload)
            return

        self._send_json(404, {"detail": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ask":
            self._send_json(404, {"detail": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8") if content_length else ""
        content_type = self.headers.get("Content-Type", "application/json")

        if content_type.startswith("application/json"):
            try:
                payload = json.loads(raw_body or "{}")
            except json.JSONDecodeError:
                self._send_json(400, {"detail": "Invalid JSON"})
                return
            question = (payload.get("question") or "").strip()
        else:
            form = parse_qs(raw_body)
            question = (form.get("question", [""])[0] or "").strip()

        if not question:
            self._send_json(400, {"detail": "Question is required"})
            return

        scene = get_latest_scene()
        question_key = _normalize_question(question)
        cache_key = f"{_scene_signature(scene)}::{question_key}"

        with _cache_lock:
            cached = _ask_cache.get(cache_key)
        if cached is not None:
            self._send_json(200, cached)
            return

        try:
            response = _client.models.generate_content(
                model="gemini-2.5-flash",
                contents=_build_caretaker_prompt(question, scene),
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You answer caretakers with short, concrete suggestions. "
                        "Focus on safety, reassurance, and immediate next steps. "
                        "Keep the response short and speak directly to the caretaker."
                    ),
                    response_mime_type="application/json",
                    response_schema=CaretakerAdvice,
                    temperature=0.2,
                ),
            )
            result = response.parsed.model_dump()
            set_latest_output(result)
            with _cache_lock:
                _ask_cache[cache_key] = result
            self._send_json(200, result)
        except Exception as error:
            self._send_json(500, {"detail": str(error)})

    def log_message(self, format: str, *args) -> None:  # silence console noise
        return


def start_http_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    return server