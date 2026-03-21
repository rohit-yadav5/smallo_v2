import time
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)


def warmup():
    """Run a dummy transcription so ONNX JIT-compiles on startup, not on first user speech."""
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence
    list(model.transcribe(dummy, language="en")[0])


def transcribe(audio_data: np.ndarray) -> tuple[str, float]:
    """
    Transcribe float32 audio at 16 kHz.

    Args:
        audio_data — float32 numpy array at 16 kHz

    Returns:
        text               — transcribed string
        transcription_secs — time Whisper spent processing
    """
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio_data, language="en")
    text = " ".join(seg.text for seg in segments).strip()
    return text, time.perf_counter() - t0
