import sounddevice as sd
import numpy as np
import webrtcvad
import collections
import requests
from faster_whisper import WhisperModel

SYSTEM_PROMPT = (
    "You are Small O, a local personal AI assistant. "
    "You are concise, practical, and honest. "
    "You help with daily tasks, coding, learning, and thinking clearly. "
    "Do not use emojis. Do not over-explain. "
    "If you do not know something, say so."
)

# =====================
# AUDIO + VAD SETTINGS
# =====================
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

vad = webrtcvad.Vad(2)

# =====================
# STT MODEL
# =====================
stt_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

def float_to_int16(audio):
    return (audio * 32768).astype(np.int16)

def run():
    while True:
        print("\n--- New Interaction ---")
        print("Speak now...")

        frames = []
        ring_buffer = collections.deque(maxlen=20)
        triggered = False

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE
        ) as stream:
            while True:
                audio, _ = stream.read(FRAME_SIZE)

                pcm = float_to_int16(audio.flatten())
                is_speech = vad.is_speech(pcm.tobytes(), SAMPLE_RATE)

                if not triggered:
                    ring_buffer.append((pcm, is_speech))
                    if sum(1 for _, s in ring_buffer if s) > 0.7 * ring_buffer.maxlen:
                        triggered = True
                        frames.extend(f for f, _ in ring_buffer)
                        ring_buffer.clear()
                else:
                    frames.append(pcm)
                    ring_buffer.append((pcm, is_speech))
                    if sum(1 for _, s in ring_buffer if not s) > 0.8 * ring_buffer.maxlen:
                        break

        print("Transcribing...")

        audio_data = np.concatenate(frames).astype(np.float32) / 32768.0
        segments, _ = stt_model.transcribe(audio_data, language="en")
        user_text = " ".join(seg.text for seg in segments)

        print("You said:", user_text)

        # =====================
        # SEND TO PHI-3 (OLLAMA)
        # =====================
        print("Thinking...")

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": f"{SYSTEM_PROMPT}\n\nUser: {user_text}\nAssistant:",
                "stream": False
            }
        )

        ai_text = response.json()["response"]
        print("AI:", ai_text)

        if user_text.strip().lower() in ["exit", "quit", "stop"]:
            print("Exiting Small O.")
            break

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nSmall O stopped by user (Ctrl+C).")
    except Exception as e:
        print("\nUnexpected error:", repr(e))