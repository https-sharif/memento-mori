from typing import List
from pydantic import BaseModel, Field


class PersonPayload(BaseModel):
    recognized: bool = False
    name: str = "Unknown"
    relationship: str = ""
    note: str = ""


class ObjectPayload(BaseModel):
    label: str


class VisionFrame(BaseModel):
    person: PersonPayload = Field(default_factory=PersonPayload)
    objects: List[ObjectPayload] = Field(default_factory=list)
    timestamp: int | None = None


class MemoryCardOutput(BaseModel):
    card_title: str = Field(description="Large, simple title text for the frontend display.")
    card_body: str = Field(description="Max 2 short sentences reinforcing context.")
    voice_guidance: str = Field(description="Text to speak to the patient.")


class CaregiverInput(BaseModel):
    message: str


class RAGChatRequest(BaseModel):
    message: str = Field(description="Caregiver/user situation or question to ground with patient memory before calling the LLM.")
    person: PersonPayload | None = None
    objects: List[ObjectPayload] = Field(default_factory=list)


class RAGDebugContext(BaseModel):
    retrieved_context: str
    llm_input: str


class RAGChatResponse(BaseModel):
    output: MemoryCardOutput
    debug: RAGDebugContext


class BehavioralAnalysis(BaseModel):
    category: str
    observed_triggers: List[str]
    clinical_rationale: str
    actionable_intervention: str
    is_crisis: bool
