from google import genai
from google.genai import types

from app.models import MemoryCardOutput, RAGChatRequest, RAGChatResponse, RAGDebugContext
from app.patient_memory import PatientMemoryStore


class RAGMemoryChatbot:
    """RAG pipeline: user/vision info -> retrieve patient memory -> LLM -> structured output."""

    def __init__(self, memory_store: PatientMemoryStore | None = None):
        self.client = genai.Client()
        self.model = "gemini-2.5-flash"
        self.memory = memory_store or PatientMemoryStore()

    def _build_query(self, request: RAGChatRequest) -> str:
        person_text = ""
        if request.person:
            person_text = (
                f"Person: {request.person.name} {request.person.relationship} "
                f"{request.person.note} recognized={request.person.recognized}."
            )
        object_text = "Objects: " + ", ".join(obj.label for obj in request.objects) if request.objects else ""
        return f"{request.message}\n{person_text}\n{object_text}".strip()

    def generate_structured_output(self, request: RAGChatRequest) -> RAGChatResponse:
        retrieval_query = self._build_query(request)
        retrieved_context = self.memory.retrieve_text(retrieval_query, top_k=5)

        llm_input = (
            "Retrieved patient memory context:\n"
            f"{retrieved_context}\n\n"
            "Live/user-provided situation:\n"
            f"{retrieval_query}"
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=llm_input,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are the structured-output generation step in a dementia memory assistant. "
                    "Use ONLY the retrieved patient memory context and live situation. "
                    "Return a MemoryCardOutput JSON object for the frontend. "
                    "Keep card text very short, gentle, reassuring, and non-clinical. "
                    "The voice guidance must speak directly to the patient. "
                    "Never mention RAG, retrieval, JSON, camera feed, analysis, or clinical labels."
                ),
                response_mime_type="application/json",
                response_schema=MemoryCardOutput,
                temperature=0.2,
            ),
        )

        return RAGChatResponse(
            output=response.parsed,
            debug=RAGDebugContext(retrieved_context=retrieved_context, llm_input=llm_input),
        )
