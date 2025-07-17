from pydub.generators import Sine
from pydub import AudioSegment

# Generate 1 second of 440Hz tone (A4)
tone = Sine(440).to_audio_segment(duration=1000)

# Save to dummy.wav
tone.export("dummy.wav", format="wav")

print("dummy.wav generated.")
