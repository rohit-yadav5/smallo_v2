"""tts/audio_utils.py — Audio encoding helpers for WebSocket delivery.

Functions
─────────
encode_pcm16(audio, sample_rate) → bytes
    Convert float32 audio to raw signed-16-bit PCM bytes.
    No header — just the raw samples.  Decodable by the Web Audio API:
      new Int16Array(buffer) then scale by 1/32768 to float32.

encode_opus(audio, sample_rate) → bytes
    Encode float32 audio as a single Opus packet via opuslib.
    Falls back to encode_pcm16 if opuslib is not installed.

chunk_audio(audio, sample_rate, chunk_ms) → list[np.ndarray]
    Split audio into equal-sized chunks of chunk_ms milliseconds.
    The last chunk is zero-padded to a full chunk if shorter.
"""
import numpy as np


def encode_pcm16(audio: np.ndarray, sample_rate: int) -> bytes:  # noqa: ARG001 (sr unused but kept for uniform signature)
    """Convert float32 audio → raw signed-16-bit PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def encode_opus(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Encode float32 audio as Opus bytes.

    Uses opuslib if available; silently falls back to encode_pcm16 otherwise.
    Opus encoder is created fresh per call (stateless helper — no streaming).

    Supported sample rates by Opus: 8000, 12000, 16000, 24000, 48000.
    If sample_rate is not in that set, the audio is returned as pcm16.
    """
    _OPUS_RATES = {8000, 12000, 16000, 24000, 48000}
    if sample_rate not in _OPUS_RATES:
        return encode_pcm16(audio, sample_rate)
    try:
        import opuslib  # type: ignore
        enc = opuslib.Encoder(sample_rate, channels=1, application=opuslib.APPLICATION_AUDIO)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        # Opus frame size must be 2.5, 5, 10, 20, 40, or 60 ms.
        # Use the full audio as one frame; clip to max 60 ms if longer.
        frame_size = min(len(audio), int(sample_rate * 0.060))
        return enc.encode(pcm16[:frame_size * 2], frame_size)
    except ImportError:
        return encode_pcm16(audio, sample_rate)
    except Exception:
        return encode_pcm16(audio, sample_rate)


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_ms: int,
) -> list[np.ndarray]:
    """
    Split audio into equal-sized chunks of chunk_ms milliseconds.

    The last chunk is zero-padded to a full chunk length so every chunk has
    exactly chunk_size samples — the frontend decoder can rely on this.

    Returns an empty list if audio is empty.
    """
    if len(audio) == 0:
        return []
    chunk_size = int(sample_rate * chunk_ms / 1000)
    if chunk_size <= 0:
        return [audio]
    # Pad to a multiple of chunk_size
    remainder = len(audio) % chunk_size
    if remainder:
        pad = chunk_size - remainder
        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]


# ── Standalone smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    SR  = 24_000
    dur = 0.1   # 100 ms tone
    t   = np.linspace(0, dur, int(SR * dur), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    pcm = encode_pcm16(tone, SR)
    print(f"encode_pcm16: {len(pcm)} bytes  (expected {len(tone)*2})")
    assert len(pcm) == len(tone) * 2, "pcm16 length mismatch"

    chunks = chunk_audio(tone, SR, chunk_ms=20)
    expected_chunk_size = int(SR * 20 / 1000)
    print(f"chunk_audio(20ms): {len(chunks)} chunks  each={expected_chunk_size} samples")
    for i, c in enumerate(chunks):
        assert len(c) == expected_chunk_size, f"chunk {i} has wrong size {len(c)}"

    opus = encode_opus(tone, SR)
    print(f"encode_opus: {len(opus)} bytes  ({'opus' if len(opus) < len(pcm) else 'pcm16 fallback'})")

    print("audio_utils smoke test PASSED")
