"""
stt/main_stt.py — Public transcribe() entry point.

Wires together:
  - engine.py   (model loading, device detection, stt_lock)
  - filters.py  (hallucination blocklist + repetition detector)

Public API is unchanged:
  transcribe(audio_data: np.ndarray) -> tuple[str, float]
  warmup() -> None

Improvements over the original base.en implementation:
  - Model:           distil-small.en  (2–3× faster, higher accuracy)
  - Device:          CUDA float16 if available, else CPU int8
  - initial_prompt:  "Small O."  (primes bot-name transcription)
  - word_timestamps: True  (collected for future word-highlight feature)
  - no_speech gate:  per-segment Python check at 0.55 in addition to
                     Whisper's own internal gate at 0.6
  - Hallucination filter: blocklist + repetition detector (filters.py)
"""
import time

import numpy as np

from stt.engine  import load_model, warmup as _engine_warmup, stt_lock, SAMPLE_RATE
from stt.filters import is_hallucination

# ── Constants ─────────────────────────────────────────────────────────────────

# Seed Whisper's context window with the bot's name so it transcribes
# "Small O" correctly instead of "smallo" / "small oh" / "small zero".
# The period trains the model that this is statement context, not mid-clause.
# Keep it short — a long prompt can bleed into short utterance transcriptions.
_INITIAL_PROMPT = "Small O."

# Peak amplitude below this value (~-46 dBFS) is definitionally inaudible.
# Reject before calling Whisper — saves the full inference time on silence.
_ENERGY_THRESHOLD = 0.005

# Whisper reports no_speech_prob per segment.  Segments above this threshold
# are discarded even if they produced text (hallucinations on background noise).
# Slightly more aggressive than Whisper's own internal 0.6 gate.
_NO_SPEECH_THRESH = 0.55

from config.limits import VAD_NO_SPEECH_THRESHOLD as _PARTIAL_NO_SPEECH_THRESH

# ── Model (loaded once at import time) ────────────────────────────────────────
_model = load_model()
if _model is None:
    raise RuntimeError("Whisper model failed to load — cannot start STT")
_stt_available: bool = True


# ── Public API ────────────────────────────────────────────────────────────────

def warmup() -> None:
    """Pre-warms CTranslate2 JIT on startup.  Called once from main.py."""
    global _stt_available
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
    with ThreadPoolExecutor(max_workers=1) as _ex:
        _fut = _ex.submit(_engine_warmup, _model)
        try:
            _fut.result(timeout=30)
        except _FutureTimeout:
            print("  [stt] warmup timed out after 30s — STT marked unavailable", flush=True)
            _stt_available = False
        except Exception as _exc:
            print(f"  [stt] warmup failed: {_exc} — STT marked unavailable", flush=True)
            _stt_available = False


def transcribe_partial(
    audio_data: np.ndarray,
) -> tuple[str, list[tuple[str, float, float]], float]:
    """Fast partial transcription for live display.  Thread-safe.

    Uses beam_size=1 (greedy, ~3× faster than transcribe()). Returns per-word
    timestamps so StreamingTranscriber can anchor its agreement algorithm on
    absolute word positions rather than text alone — this prevents jitter when
    Whisper rewrites punctuation or capitalisation across consecutive runs.

    Returns:
        (text, [(word_str, start_secs, probability), ...], secs)
        text and words list are both "" / [] if audio is silent or hallucination.
    """
    if not _stt_available:
        return "", [], 0.0
    with stt_lock:
        return _transcribe_partial(audio_data)


def transcribe(audio_data: np.ndarray) -> tuple[str, float]:
    """Transcribe float32 audio at 16 kHz.  Thread-safe.

    Args:
        audio_data: float32 numpy array at SAMPLE_RATE (16 000 Hz).

    Returns:
        (text, transcription_secs) — text is "" if the audio is too quiet,
        all segments fail the no_speech gate, or the result is a known
        hallucination.
    """
    if not _stt_available:
        return "", 0.0
    with stt_lock:
        return _transcribe(audio_data)


# ── Internal ──────────────────────────────────────────────────────────────────

def _transcribe(audio_data: np.ndarray) -> tuple[str, float]:
    """Inner transcription — caller must hold stt_lock."""
    audio_data = audio_data.copy()
    np.clip(audio_data, -1.0, 1.0, out=audio_data)

    # ── Layer 1: Amplitude gate ───────────────────────────────────────────
    # Cheap numpy max — avoids the full Whisper inference on near-silence.
    peak = float(np.abs(audio_data).max())
    if peak < _ENERGY_THRESHOLD:
        return "", 0.0

    # Normalise to 90 % of full scale after the energy gate — this way
    # near-silent frames are not amplified to full-scale and passed to Whisper.
    audio_data = (audio_data / peak * 0.9).astype(np.float32)

    # ── Layer 2: Whisper inference ────────────────────────────────────────
    t0 = time.perf_counter()
    segments_gen, _ = _model.transcribe(
        audio_data,
        language                   = "en",
        beam_size                  = 5,
        best_of                    = 1,
        temperature                = 0.0,    # greedy / deterministic
        condition_on_previous_text = False,  # prevent prior-turn bleeding
        vad_filter                 = False,  # Silero VAD already handled this
        no_speech_threshold        = 0.6,    # Layer 2: Whisper's internal gate
        initial_prompt             = _INITIAL_PROMPT,
        word_timestamps            = True,   # enables seg.words for future use
    )

    # ── Layer 3: Per-segment no_speech_prob gate ──────────────────────────
    # Inspect each yielded segment's probability directly — slightly more
    # aggressive than Whisper's own 0.6 internal gate.
    kept_texts: list[str] = []

    for seg in segments_gen:
        if seg.no_speech_prob > _NO_SPEECH_THRESH:
            continue   # discard segment — model is unsure speech is present
        kept_texts.append(seg.text)
        # seg.words is list[Word] with .word, .start, .end, .probability
        # Available for future word-highlight frontend feature.

    transcription_secs = time.perf_counter() - t0
    text = " ".join(kept_texts).strip()

    # ── Layers 4 + 5: Hallucination filter ───────────────────────────────
    if is_hallucination(text):
        return "", transcription_secs

    return text, transcription_secs


def _transcribe_partial(audio_data: np.ndarray) -> tuple[str, list[tuple[str, float, float]], float]:
    """Inner partial transcription — caller must hold stt_lock."""
    audio_data = audio_data.copy()
    np.clip(audio_data, -1.0, 1.0, out=audio_data)

    # ── Layer 1: Amplitude gate ───────────────────────────────────────────
    peak = float(np.abs(audio_data).max())
    if peak < _ENERGY_THRESHOLD:
        return "", [], 0.0
    audio_data = (audio_data / peak * 0.9).astype(np.float32)

    # ── Layer 2: Whisper inference (greedy — beam_size=1) ─────────────────
    t0 = time.perf_counter()
    segments_gen, _ = _model.transcribe(
        audio_data,
        language                   = "en",
        beam_size                  = 1,          # greedy: ~3× faster on CPU
        best_of                    = 1,
        temperature                = 0.0,
        condition_on_previous_text = False,
        vad_filter                 = False,
        no_speech_threshold        = _PARTIAL_NO_SPEECH_THRESH,
        initial_prompt             = _INITIAL_PROMPT,
        word_timestamps            = True,
    )

    # ── Layer 3: Per-segment no_speech_prob gate ──────────────────────────
    kept_texts: list[str]                      = []
    all_words:  list[tuple[str, float, float]] = []

    for seg in segments_gen:
        if seg.no_speech_prob > _PARTIAL_NO_SPEECH_THRESH:
            continue
        kept_texts.append(seg.text)
        if seg.words:
            for w in seg.words:
                all_words.append((w.word, w.start, w.probability))

    secs = time.perf_counter() - t0
    text = " ".join(kept_texts).strip()

    # ── Layers 4 + 5: Hallucination filter ───────────────────────────────
    if is_hallucination(text):
        return "", [], secs

    return text, all_words, secs
