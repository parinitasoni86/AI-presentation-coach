# app/models.py

from pydantic import BaseModel
from typing import List, Optional

class ScriptIn(BaseModel):
    script: str
    tone: str = "formal"
    length_minutes: int = 2

class FillerOut(BaseModel):
    filler_words: List[str]
    cleaned_script: str

class FeedbackOut(BaseModel):
    suggestions: List[str]
    score: Optional[float] = None

class PitchOut(BaseModel):
    pitch: str
