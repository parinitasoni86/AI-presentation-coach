# app/services.py

import os
import whisper
import speech_recognition as sr
import google.generativeai as genai
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

FILLERS = {"um", "uh", "like", "you", "know", "so", "actually", "basically"}
whisper_model = whisper.load_model("base")

def gemini_chat(prompt: str, **kwargs) -> str:
    rsp = model.generate_content(prompt, **kwargs)
    return rsp.text.strip()

def detect_filler_words(script: str):
    words = script.split()
    found = [w for w in words if w.lower().strip(",.") in FILLERS]
    cleaned = " ".join(w for w in words if w.lower().strip(",.") not in FILLERS)
    return found, cleaned

def generate_pitch(script: str, tone: str, length_minutes: int) -> str:
    prompt = (f"Write a {length_minutes}-minute elevator pitch in a "
              f"{tone} tone based on these key points:\n{script}")
    return gemini_chat(prompt, generation_config={"max_output_tokens": 512})

def give_feedback(script: str):
    prompt = ("Provide 5 bullet-point suggestions to improve clarity, structure, "
              "storytelling, and timing for this script. End with 'Score: X/10'.\n\n" + script)
    txt = gemini_chat(prompt)
    lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
    tips = [l for l in lines if not l.lower().startswith("score")]
    score = next((float(l.split()[1].split('/')[0]) for l in lines
                  if l.lower().startswith("score")), None)
    return tips[:5], score

def transcribe_audio_to_text(file_path: str) -> str:
    result = whisper_model.transcribe(file_path, fp16=False)
    return result["text"].strip()
