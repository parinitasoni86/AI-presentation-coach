# main.py - AI Presentation Coach backend (Production-ready)

import os, io, logging, tempfile
import whisper
import speech_recognition as sr
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi import Request
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

import logging
from fastapi.logger import logger

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------
# Load environment variables & logging setup
# ------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Configure Gemini with secure key from .env
# ------------------------------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY:", "AIzaSyD_1yMw8C-vA1VrtUMljbBRlEDPyFB_kt4")

if not API_KEY:
    logging.error("GOOGLE_API_KEY not set in .env")
    raise EnvironmentError("Missing GOOGLE_API_KEY")

genai.configure(api_key="AIzaSyD_1yMw8C-vA1VrtUMljbBRlEDPyFB_kt4")

model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# ------------------------------------------------------
# FastAPI init
# ------------------------------------------------------
app = FastAPI(title="AI Presentation Coach (Production Ready)")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific frontend URLs like "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@limiter.limit("10/minute")  # Limit to 5 requests per minute per IP
@app.post("/detect_fillers", response_model=FillerOut)
def detect_fillers(request: Request, item: ScriptIn):
    logger = logging.getLogger("detect_fillers")
    try:
        words = item.script.split()
        found = [w for w in words if w.lower().strip(",.") in FILLERS]
        cleaned = " ".join(w for w in words if w.lower().strip(",.") not in FILLERS)

        logger.info(f"Detected {len(found)} filler words")
        return FillerOut(filler_words=found, cleaned_script=cleaned)
    
    except Exception as e:
        logger.exception("Error in detecting fillers")
        raise HTTPException(500, detail="Error detecting filler words")


@limiter.limit("5/minute")
@app.post("/generate_pitch", response_model=PitchOut)
def generate_pitch(request: Request, item: ScriptIn):
    logger = logging.getLogger("generate_pitch")
    try:
        if item.length_minutes not in (2, 3, 4):
            logger.warning(f"Invalid pitch length: {item.length_minutes}")
            raise HTTPException(400, detail="length_minutes must be 2, 3, or 4")

        prompt = (f"Write a {item.length_minutes}-minute elevator pitch in a "
                  f"{item.tone} tone based on these key points:\n{item.script}")
        pitch = gemini_chat(prompt, generation_config={"max_output_tokens": 512})
        logger.info("Pitch generated successfully")

        return PitchOut(pitch=pitch)

    except Exception as e:
        logger.exception("Error generating pitch")
        raise HTTPException(500, detail="Error generating pitch")


@limiter.limit("5/minute")
@app.post("/script_coach", response_model=FeedbackOut)
def script_coach(request: Request, item: ScriptIn):
    logger = logging.getLogger("script_coach")
    try:
        prompt = ("Provide 5 bullet-point suggestions to improve clarity, structure, "
                  "storytelling, and timing for this script. End with 'Score: X/10'.\n\n"
                  f"{item.script}")
        txt = gemini_chat(prompt)
        lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
        tips = [l for l in lines if not l.lower().startswith("score")]
        score = next((float(l.split()[1].split('/')[0]) for l in lines
                      if l.lower().startswith("score")), None)

        logger.info(f"Coaching completed with score: {score}")
        return FeedbackOut(suggestions=tips[:5], score=score)

    except Exception as e:
        logger.exception("Error providing script feedback")
        raise HTTPException(500, detail="Error providing script feedback")



@limiter.limit("3/minute")
@app.post("/transcribe_audio", response_model=ScriptIn)
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    logger = logging.getLogger("transcribe_audio")
    
    # Validate file type
    if not file.content_type.startswith("audio/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(400, detail="Please upload an audio file (wav/mp3/flac/m4a…)")

    # Save upload to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            tmp_path = tmp.name
            logger.info(f"Saved uploaded audio to {tmp_path}")
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(500, detail="Failed to save uploaded file")

    # Transcribe using Whisper
    try:
        result = whisper_model.transcribe(tmp_path, fp16=False)
        text = result["text"].strip()
        logger.info("Transcription successful")
        return ScriptIn(script=text)
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(500, detail="Transcription failed")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.info("Temp file deleted")
        except Exception as e:
            logger.warning(f"Could not delete temp file {tmp_path}: {e}")

