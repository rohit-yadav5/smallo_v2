import time
import sounddevice as sd
import numpy as np
import webrtcvad
import collections
from faster_whisper import WhisperModel

# =====================
# AUDIO SETTINGS
# =====================
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# Less aggressive VAD for reliability
vad = webrtcvad.Vad(1)

# =====================
# STT MODEL
# =====================
model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)


def warmup():
    """Run a dummy transcription so ONNX JIT-compiles on startup, not on first user speech."""
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence
    list(model.transcribe(dummy, language="en")[0])   # consume the generator


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (audio * 32768).astype(np.int16)


def listen() -> tuple[str, float, float]:
    """
    Listen for speech and transcribe it.

    Returns:
        text            — transcribed string
        recording_secs  — time from first voiced frame to end-of-speech detection
        transcription_secs — time Whisper spent processing the audio
    """
    frames = []
    ring_buffer = collections.deque(maxlen=20)
    triggered = False
    recording_start: float | None = None

    print("\nSpeak now...")

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
                voiced = sum(1 for _, s in ring_buffer if s)

                if voiced > 0.6 * ring_buffer.maxlen:
                    triggered = True
                    recording_start = time.perf_counter()
                    frames.extend(f for f, _ in ring_buffer)
                    ring_buffer.clear()
            else:
                frames.append(pcm)
                ring_buffer.append((pcm, is_speech))
                unvoiced = sum(1 for _, s in ring_buffer if not s)

                if unvoiced > 0.8 * ring_buffer.maxlen:
                    break

    recording_secs = (time.perf_counter() - recording_start) if recording_start else 0.0

    if not frames:
        print("No speech detected.")
        return "", 0.0, 0.0

    audio_data = np.concatenate(frames).astype(np.float32) / 32768.0

    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio_data, language="en")
    text = " ".join(seg.text for seg in segments).strip()
    transcription_secs = time.perf_counter() - t0

    return text, recording_secs, transcription_secs
