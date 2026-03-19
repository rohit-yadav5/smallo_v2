import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 5  # seconds

print("Listening...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)
sd.wait()

audio = audio.flatten()

model = WhisperModel(
    "base",
    device="cpu",        # change to "cuda" later if needed
    compute_type="int8"
)

segments, _ = model.transcribe(audio, language="en")

text = " ".join([seg.text for seg in segments])
print("You said:", text)