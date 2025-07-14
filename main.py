# main.py - AI Presentation Coach backend (Production-ready)

import os, io, logging, tempfile
import whisper
import speech_recognition as sr
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------------------------------------------
# Load environment variables & logging setup
# ------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.info("Backend starting up...")

# ------------------------------------------------------
# Configure Gemini with secure key from .env
# ------------------------------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY:", "AIzaSyBNLzAJTkakn66hQdUhvpHYJJA3Ei4wezQ")

if not API_KEY:
    logging.error("GOOGLE_API_KEY not set in .env")
    raise EnvironmentError("Missing GOOGLE_API_KEY")

genai.configure(api_key="AIzaSyBNLzAJTkakn66hQdUhvpHYJJA3Ei4wezQ")

model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# ------------------------------------------------------
# FastAPI init
# ------------------------------------------------------
app = FastAPI(title="AI Presentation Coach (Production Ready)")

# ------------------------------------------------------
# Pydantic models
# ------------------------------------------------------
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

# ------------------------------------------------------
# Constants & reusable logic
# ------------------------------------------------------
FILLERS = {"um", "uh", "like", "you", "know", "so", "actually", "basically"}
whisper_model = whisper.load_model("base")  # Options: tiny/base/small/medium/large


def gemini_chat(prompt: str, **kwargs) -> str:
    try:
        rsp = model.generate_content(prompt, **kwargs)
        return rsp.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        raise HTTPException(500, "Gemini model error")

# ------------------------------------------------------
# Endpoints
# ------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "running", "msg": "Backend ready"}

@app.post("/detect_fillers", response_model=FillerOut)
def detect_fillers(item: ScriptIn):
    words = item.script.split()
    found = [w for w in words if w.lower().strip(",.") in FILLERS]
    cleaned = " ".join(w for w in words if w.lower().strip(",.") not in FILLERS)
    return FillerOut(filler_words=found, cleaned_script=cleaned)

@app.post("/generate_pitch", response_model=PitchOut)
def generate_pitch(item: ScriptIn):
    if item.length_minutes not in (2, 3, 4):
        raise HTTPException(400, "length_minutes must be 2, 3, or 4")

    prompt = (
        f"Write a {item.length_minutes}-minute elevator pitch in a "
        f"{item.tone} tone based on these key points:\n{item.script}"
    )
    pitch = gemini_chat(prompt, generation_config={"max_output_tokens": 512})
    return PitchOut(pitch=pitch)

@app.post("/script_coach", response_model=FeedbackOut)
def script_coach(item: ScriptIn):
    prompt = (
        "Provide 5 bullet-point suggestions to improve clarity, structure, "
        "storytelling, and timing for this script. End with 'Score: X/10'.\n\n"
        f"{item.script}"
    )
    txt = gemini_chat(prompt)
    lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
    tips = [l for l in lines if not l.lower().startswith("score")]
    score = next(
        (float(l.split()[1].split("/")[0]) for l in lines if l.lower().startswith("score")),
        None
    )
    return FeedbackOut(suggestions=tips[:5], score=score)

@app.post("/transcribe_audio", response_model=ScriptIn)
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Please upload a valid audio file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(tmp_path, fp16=False)
        text = result["text"].strip()
        return ScriptIn(script=text)
    except Exception as e:
        logging.error(f"Whisper transcription error: {e}")
        raise HTTPException(500, "Audio transcription failed")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
