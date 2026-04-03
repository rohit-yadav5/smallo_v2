"""
stt/engine.py — Device detection, model loading, and warmup helper.

faster-whisper uses ctranslate2 internally.
ctranslate2 only supports 'cpu' and 'cuda' — MPS/Metal is NOT supported
regardless of torch.backends.mps.is_available(). On Apple Silicon the
device chain is: CUDA (if NVIDIA GPU present) → CPU.
"""
import threading

import numpy as np
import torch
from faster_whisper import WhisperModel

# ── Model selection ───────────────────────────────────────────────────────────
# distil-small.en is ~2–3× faster than base.en on CPU int8 and more accurate
# (distilled from Whisper large-v3).  faster-whisper ≥1.2 recognises the alias
# "distil-small.en" and maps it to Systran/faster-distil-whisper-small.en.
# Falls back to "small.en" if the primary download fails (e.g. offline).
PRIMARY_MODEL  = "distil-small.en"
FALLBACK_MODEL = "small.en"
SAMPLE_RATE    = 16_000


def _select_device() -> tuple[str, str]:
    """Return (device, compute_type) for ctranslate2.

    MPS is explicitly skipped — ctranslate2 4.x has no Metal backend.
    """
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


_DEVICE, _COMPUTE_TYPE = _select_device()

# Exported lock — all callers (main_stt + streaming) share the same mutex so
# the underlying CTranslate2 ONNX session is never called concurrently.
stt_lock = threading.Lock()


def load_model() -> WhisperModel:
    """Load distil-small.en, falling back to small.en on failure.

    Prints the model name and device to the startup log so the user can
    see what was loaded.
    """
    for name in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            m = WhisperModel(name, device=_DEVICE, compute_type=_COMPUTE_TYPE)
            print(
                f"  [stt/engine] loaded '{name}' on {_DEVICE} ({_COMPUTE_TYPE})",
                flush=True,
            )
            return m
        except Exception as exc:
            print(f"  [stt/engine] failed to load '{name}': {exc}", flush=True)
    raise RuntimeError("STT: all model load attempts failed")


def warmup(model: WhisperModel) -> None:
    """Run one dummy transcription to trigger CTranslate2 JIT compilation.

    Called once at startup so the first real utterance doesn't pay the
    ONNX compilation cost.
    """
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    with stt_lock:
        list(model.transcribe(dummy, language="en")[0])
