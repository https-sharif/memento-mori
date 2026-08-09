from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryCard(BaseModel):
    card_title: str = Field(description="Large, simple title text for the frontend display (e.g., 'Sarah (Daughter)').")
    card_body: str = Field(description="Max 2 short sentences reinforcing context (e.g., 'She visits every weekend. She loves you.').")
    voice_guidance: str = Field(description="The exact text to convert to audio. Must be short, exceptionally gentle, comforting, and spoken directly to the patient or caretaker.")


class AskRequest(BaseModel):
    question: str


class CaregiverInput(BaseModel):
    message: str


class BehavioralAnalysis(BaseModel):
    category: str = Field(description="Categorization of behavior: e.g., Sundowning, Wandering, Aggression, Confusion")
    observed_triggers: list[str] = Field(description="Possible environmental, physical, or temporal triggers extracted from text.")
    clinical_rationale: str = Field(description="Brief neuro-clinical context explaining why the patient is exhibiting this specific behavior.")
    actionable_intervention: str = Field(description="Direct, actionable, non-pharmacological step for the caregiver to de-escalate.")
    is_crisis: bool = Field(description="Set to True ONLY if immediate physical danger or medical emergency is indicated.")
