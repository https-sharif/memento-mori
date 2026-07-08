from __future__ import annotations

from pydantic import BaseModel, Field


class CaretakerAdvice(BaseModel):
    card_title: str = Field(description="Short title for the advice card.")
    card_body: str = Field(description="Short advice for the caretaker.")
    voice_guidance: str = Field(description="Short sentence to speak aloud to the caretaker.")