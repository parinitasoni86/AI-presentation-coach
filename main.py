# main.py - AI Presentation Coach backend (Production-ready)

import os, io, logging, tempfile
import whisper
import aiofiles
import speech_recognition as sr
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi import Request
from pydantic import BaseModel, Field, constr, conint
from typing import List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.exceptions import HTTPException as FastAPIHTTPException


from loguru import logger
import time
import uuid
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Clear default handlers if needed
logger.remove()

# Add log file and console logging
logger.add("logs/server.log", rotation="1 MB", retention="7 days", enqueue=True, level="INFO")


import logging
from fastapi.logger import logger as fastapi_logger 

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime
import uuid
import os

TRANSCRIPTS_DIR = "transcripts"
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

def save_transcript_to_file(text: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    filename = f"transcript_{timestamp}_{uid}.txt"
    filepath = os.path.join(TRANSCRIPTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filename

# ------------------------------------------------------
# Load environment variables & logging setup
# ------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
python_logger = logging.getLogger(__name__)

from hashlib import sha256
from functools import lru_cache

gemini_cache = {}

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
    allow_origins=["http://localhost:3000"],  # ✅ Replace with your real frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"🔵 [ReqID {request_id}] Start request: {request.method} {request.url}")
    
    response = await call_next(request)

    logger.info(f"✅ [ReqID {request_id}] End request: {response.status_code}")
    return response

from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi import status

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, FastAPIHTTPException):
        raise exc  # Let FastAPI handle it normally
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )

# ------------------------------------------------------
# Pydantic models
# ------------------------------------------------------
class ScriptIn(BaseModel):
    script: constr(min_length=10, max_length=5000)
    tone: constr(to_lower=True, pattern="^(formal|casual|motivational|humorous)$") = Field(...)
    length_minutes: conint(gt=0, le=10) = Field(..., description="Presentation length in minutes (1-10)")

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
        python_logger.error(f"Gemini API error: {e}")  # <- this line
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
        cache_key = sha256(f"fillers:{item.script}".encode()).hexdigest()
        if cache_key in gemini_cache:
            logger.info("🎯 Returning cached filler analysis")
            return gemini_cache[cache_key]

        words = item.script.split()
        found = [w for w in words if w.lower().strip(",.") in FILLERS]
        cleaned = " ".join(w for w in words if w.lower().strip(",.") not in FILLERS)

        result = FillerOut(filler_words=found, cleaned_script=cleaned)
        gemini_cache[cache_key] = result

        logger.info(f"✅ Detected {len(found)} filler words")
        return result

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

        cache_key = sha256(f"pitch:{item.length_minutes}:{item.tone}:{item.script}".encode()).hexdigest()
        if cache_key in gemini_cache:
            logger.info("🎯 Returning cached pitch result")
            return PitchOut(pitch=gemini_cache[cache_key])

        prompt = (f"Write a {item.length_minutes}-minute elevator pitch in a "
                  f"{item.tone} tone based on these key points:\n{item.script}")
        pitch = gemini_chat(prompt, generation_config={"max_output_tokens": 512})
        gemini_cache[cache_key] = pitch

        logger.info("Pitch generated and cached successfully")
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

        cache_key = sha256(f"coach:{item.script}".encode()).hexdigest()
        if cache_key in gemini_cache:
            logger.info("🎯 Returning cached coaching feedback")
            txt = gemini_cache[cache_key]
        else:
            txt = gemini_chat(prompt)
            gemini_cache[cache_key] = txt

        lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
        tips = [l for l in lines if not l.lower().startswith("score")]
        score = next((float(l.split()[1].split('/')[0]) for l in lines
                      if l.lower().startswith("score")), None)

        logger.info(f"✅ Coaching done with score: {score}")
        return FeedbackOut(suggestions=tips[:5], score=score)

    except Exception as e:
        logger.exception("Error providing script feedback")
        raise HTTPException(500, detail="Error providing script feedback")


@limiter.limit("3/minute")
@app.post("/transcribe_audio", response_model=ScriptIn)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: str = "base",
    language: Optional[str] = None
):
    if model not in ("tiny", "base", "small", "medium", "large"):
        raise HTTPException(400, "Invalid model. Choose from: tiny, base, small, medium, large.")
    
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/flac"):
        raise HTTPException(400, "Supported formats: WAV / FLAC")

    tmp_dir = tempfile.gettempdir()
    tmp_in_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.input")
    tmp_wav_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")

    try:
        # Save uploaded file asynchronously
        async with aiofiles.open(tmp_in_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Convert to WAV
        sound = AudioSegment.from_file(tmp_in_path)
        sound.export(tmp_wav_path, format="wav")

        # Load Whisper model dynamically
        whisper_model = whisper.load_model(model)

        result = whisper_model.transcribe(tmp_wav_path, language=language)
        text = result.get("text", "").strip()

        filename = save_transcript_to_file(text)
        return {"script": text, "saved_as": filename}

    finally:
        for path in [tmp_in_path, tmp_wav_path]:
            if os.path.exists(path):
                os.unlink(path)


logger.info("🚀 Logging is set up and application is starting...")

