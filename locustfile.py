from locust import HttpUser, task, between
import random
import json
import os

class PresentationCoachUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://127.0.0.1:8000"  # ✅ Set your backend URL here

    # -------------- Shared Sample Data ------------------

    scripts = [
        "Today I'm going to tell you about our startup idea focused on sustainable packaging.",
        "Our project is about using AI to streamline financial reporting for small businesses.",
        "In this pitch, I want to talk about the importance of inclusive design in tech products.",
    ]

    tones = ["formal", "persuasive", "enthusiastic"]

    # -------------- Individual Tasks ---------------------


    @task(1)
    def detect_fillers(self):
        data = {
            "script": random.choice(self.scripts),
            "tone": random.choice(self.tones),
            "length_minutes": random.choice([2, 3, 4])
        }
        self.client.post("/detect_fillers", json=data)

    @task(1)
    def generate_pitch(self):
        data = {
            "script": random.choice(self.scripts),
            "tone": random.choice(self.tones),
            "length_minutes": random.choice([2, 3, 4])
        }
        self.client.post("/generate_pitch", json=data)

    @task(1)
    def script_coach(self):
        data = {
            "script": random.choice(self.scripts),
            "tone": random.choice(self.tones),
            "length_minutes": random.choice([2, 3, 4])
        }
        self.client.post("/script_coach", json=data)

    @task(1)
    def transcribe_audio(self):
        file_path = os.path.join(os.path.dirname(__file__), "dummy.wav")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                files = {"file": ("dummy.wav", f, "audio/wav")}
                self.client.post("/transcribe_audio", files=files)
        else:
            print("dummy.wav not found. Please place it in the same directory as locustfile.py.")
