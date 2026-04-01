import time
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000

# "base.en" is the English-only base model — ~3× more accurate than "tiny"
# with only a modest latency increase (~0.5–1 s extra on CPU with int8).
model = WhisperModel(
    "base.en",
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
        text               — transcribed string (empty if audio is too quiet)
        transcription_secs — time Whisper spent (0.0 if energy gate rejects it)
    """
    audio_data = audio_data.copy()
    np.clip(audio_data, -1.0, 1.0, out=audio_data)

    # ── Energy gate ────────────────────────────────────────────────────────
    # Reject near-silent audio before passing to Whisper.  Without this, Whisper
    # hallucinates filler words ("you", "thank you", "bye") on silence/noise.
    # Peak < 0.005 is below ~-46 dBFS — too quiet to be real speech.
    peak = float(np.abs(audio_data).max())
    if peak < 0.005:
        return "", 0.0

    # ── Normalise once per utterance ──────────────────────────────────────
    # Scale to 90 % of full scale so Whisper always sees a well-levelled signal
    # regardless of mic gain.  We only do this after the energy gate so that
    # near-silent frames are not amplified to full-scale (which causes hallucinations).
    audio_data = (audio_data / peak * 0.9).astype(np.float32)

    t0 = time.perf_counter()
    segments, _ = model.transcribe(
        audio_data,
        language        = "en",
        beam_size       = 5,      # 5 beams → noticeably better accuracy vs beam=1
        best_of         = 1,      # only meaningful when temperature > 0; set to 1
        temperature     = 0.0,    # greedy / deterministic — best for clean voice
        condition_on_previous_text = False,   # prevent prior turns biasing output
        # vad_filter is intentionally disabled.  Our Silero VAD already provides
        # clean speech-only segments with 200 ms pre-speech padding.  Whisper's
        # own VAD (also Silero-based) double-filters and can clip the start of
        # short words, causing words like "yes"/"ok" to be mis-transcribed.
        vad_filter          = False,
        no_speech_threshold = 0.6,    # suppress hallucinations (Whisper's own gate)
        word_timestamps     = False,  # not needed, saves ~10 % inference time
    )
    text = " ".join(seg.text for seg in segments).strip()
    return text, time.perf_counter() - t0
