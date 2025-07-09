# main.py  –  AI Presentation Coach backend (Gemini Pro)

import os, io
import speech_recognition as sr
import tempfile
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from pydub import AudioSegment

# -- configure Gemini -------------------------------------------------
genai.configure(api_key=os.getenv("AIzaSyAK5vcQHVGE2Xp8PAPIOId2eqx8R2m3ISY"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# -- FastAPI init -----------------------------------------------------
app = FastAPI(title="AI Presentation Coach (Gemini)")

# -- pydantic models --------------------------------------------------
class ScriptIn(BaseModel):
    script: str
    tone: str = "formal"          # persuasive / enthusiastic …
    length_minutes: int = 2       # 2–4

class FillerOut(BaseModel):
    filler_words: List[str]
    cleaned_script: str

class FeedbackOut(BaseModel):
    suggestions: List[str]
    score: Optional[float] = None

class PitchOut(BaseModel):
    pitch: str

# -- helper -----------------------------------------------------------
FILLERS = {"um","uh","like","you","know","so","actually","basically"}

def gemini_chat(prompt: str, **kwargs) -> str:
    """Simple wrapper around Gemini Pro."""
    rsp = model.generate_content(prompt, **kwargs)
    return rsp.text.strip()

# -- endpoints --------------------------------------------------------
@app.get("/")
async def root():
    return {"status":"running","msg":"Gemini backend ready"}

@app.post("/detect_fillers", response_model=FillerOut)
def detect_fillers(item: ScriptIn):
    words = item.script.split()
    found = [w for w in words if w.lower().strip(",.") in FILLERS]
    cleaned = " ".join(w for w in words if w.lower().strip(",.") not in FILLERS)

    return FillerOut(filler_words=found, cleaned_script=cleaned)

@app.post("/generate_pitch", response_model=PitchOut)
def generate_pitch(item: ScriptIn):
    if item.length_minutes not in (2,3,4):
        raise HTTPException(400,"length_minutes must be 2,3,4")
    prompt = (f"Write a {item.length_minutes}-minute elevator pitch in a "
              f"{item.tone} tone based on these key points:\n{item.script}")
    pitch = gemini_chat(prompt, generation_config={"max_output_tokens":512})
    return PitchOut(pitch=pitch)

@app.post("/script_coach", response_model=FeedbackOut)
def script_coach(item: ScriptIn):
    prompt = ("Provide 5 bullet-point suggestions to improve clarity, structure, "
              "storytelling, and timing for this script. End with 'Score: X/10'.\n\n"
              f"{item.script}")
    txt = gemini_chat(prompt)
    lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
    tips  = [l for l in lines if not l.lower().startswith("score")]
    score = next((float(l.split()[1].split('/')[0]) for l in lines
                  if l.lower().startswith("score")), None)
    return FeedbackOut(suggestions=tips[:5], score=score)


# @app.post("/transcribe_audio", response_model=ScriptIn)
# async def transcribe_audio(file: UploadFile = File(...)):
#     if file.content_type not in ("audio/wav", "audio/x-wav", "audio/flac"):
#         raise HTTPException(400, "Supported formats: WAV / FLAC")

#     tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".input")
#     tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

#     try:
#         # Write uploaded file to temp
#         tmp_in.write(await file.read())
#         tmp_in.close()  # Ensure it's flushed and closed

#         # Convert audio to WAV
#         sound = AudioSegment.from_file(tmp_in.name)
#         sound.export(tmp_wav.name, format="wav")
#         tmp_wav.close()  # Close the wav file too

#         recog = sr.Recognizer()
#         with sr.AudioFile(tmp_wav.name) as src:
#             audio = recog.record(src)
#             text = recog.recognize_google(audio)

#         return ScriptIn(script=text)

#     finally:
#         # Now safe to delete
#         if os.path.exists(tmp_in.name):
#             os.unlink(tmp_in.name)
#         if os.path.exists(tmp_wav.name):
#             os.unlink(tmp_wav.name)

import whisper, tempfile, os
from fastapi import UploadFile, File, HTTPException

# Load the Whisper model once at startup (outside the endpoint)
whisper_model = whisper.load_model("base")   # "tiny"|"base"|"small"|"medium"|"large"

@app.post("/transcribe_audio", response_model=ScriptIn)
async def transcribe_audio(file: UploadFile = File(...)):
    # Accept common audio types – Whisper can handle many formats directly
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Please upload an audio file (wav/mp3/flac/m4a…).")

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        tmp_path = tmp.name

    try:
        # Whisper will load/convert internally; no need for pydub
        result = whisper_model.transcribe(tmp_path, fp16=False)   # fp16=False for CPU
        text   = result["text"].strip()

        return ScriptIn(script=text)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

