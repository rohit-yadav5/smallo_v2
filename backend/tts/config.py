"""tts/config.py — TTS configuration (Kokoro ONNX engine).

Every value has a hardcoded default that matches the original behaviour.
All values are overridable via environment variable — useful for testing
a different voice, adjusting speed, or tuning latency without touching code.

Usage
─────
    from tts.config import TTS_CONFIG

    model = Kokoro(TTS_CONFIG.kokoro_model_path, TTS_CONFIG.kokoro_voices_path)

Environment variables
─────────────────────
    --- Kokoro ---
    TTS_KOKORO_MODEL_PATH   Path to kokoro-v1.0.onnx (default: kokoro-models/ in project root)
    TTS_KOKORO_VOICES_PATH  Path to voices-v1.0.bin  (default: kokoro-models/ in project root)
    TTS_KOKORO_VOICE        Voice name, e.g. af_sarah, bf_emma (default: af_sarah)
    TTS_KOKORO_SPEED        Speech speed multiplier; 1.0 = normal (default: 1.1)
    TTS_SAMPLE_RATE         Native output sample rate in Hz (default: 24000 for Kokoro)

    --- Chunking ---
    TTS_SENTENCE_SPLIT      Regex pattern used to split LLM text into sentences
    TTS_MAX_BUFFER_CHARS    Flush buffer as a chunk when it exceeds this many chars (no-split guard)
    TTS_MIN_CHUNK_WORDS     Hold chunks shorter than this many words and merge with next
    TTS_SYNTHESIS_RETRY_DELAY  Seconds to wait before retrying a failed synthesis

    --- Timing / queue ---
    LLM_TOKEN_TIMEOUT_S     Seconds to wait for first LLM token before aborting
    TTS_TOKEN_QUEUE_SIZE    Max tokens buffered between LLM producer and TTS consumer
    TTS_POLL_INTERVAL_S     How often (seconds) the playback loop checks for interrupts
    TTS_PRODUCER_RETRY_S    Timeout (seconds) on each queue.put retry in the producer
    TTS_CONSUMER_POLL_S     Timeout (seconds) on each queue.get in the consumer loop
    TTS_SENTINEL_RETRIES    Max retries to deliver the None sentinel when queue is full
    TTS_HEARTBEAT_INTERVAL_S Seconds between "waiting for first token..." log lines
    TTS_LOG_SENTENCE_CHARS  Max characters of a sentence shown in the synth log line

    --- Remote audio (WebSocket streaming) ---
    TTS_REMOTE_AUDIO        "true" to stream audio to WS clients (default: false)
    TTS_AUDIO_FORMAT        "pcm16" or "opus" (default: pcm16; opus requires opuslib)
    TTS_AUDIO_CHUNK_MS      Duration per WS audio chunk in ms (default: 20)
    TTS_MONITOR_LOCALLY     "true" to also play locally when remote_audio is on (default: false)

    --- Local playback resampling ---
    TTS_DEVICE_SAMPLE_RATE  Native sample rate of the output device (Hz).
                            Auto-detected from sounddevice at startup.
                            Override if auto-detection picks the wrong device
                            (e.g. HDMI instead of built-in speakers).
                            macOS built-in speakers are typically 44100 Hz.
"""
import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def _get_device_sample_rate() -> int:
    """Query the default output device native sample rate at import time."""
    try:
        import sounddevice as sd
        dev = sd.query_devices(sd.default.device['output'])
        return int(dev['default_samplerate'])
    except Exception:
        return 44100  # safe macOS fallback


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class TTSConfig:
    # ── Kokoro settings ───────────────────────────────────────────────────────
    # Paths default to the kokoro-models/ directory at the project root.
    # The path calculation assumes this file lives at backend/tts/config.py.
    kokoro_model_path: str = field(default_factory=lambda: _env_str(
        "TTS_KOKORO_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "kokoro-models", "kokoro-v1.0.onnx"),
    ))
    kokoro_voices_path: str = field(default_factory=lambda: _env_str(
        "TTS_KOKORO_VOICES_PATH",
        os.path.join(os.path.dirname(__file__), "kokoro-models", "voices-v1.0.bin"),
    ))
    # Voice name.  af_sarah chosen as default: American female, RTF≈0.26 (2× faster
    # than af_heart RTF≈0.52) based on benchmarking against af_heart and bf_emma.
    kokoro_voice: str   = field(default_factory=lambda: _env_str("TTS_KOKORO_VOICE", "af_sarah"))
    # Speed multiplier passed to Kokoro.create(speed=...).  1.2 = noticeably faster
    # but still natural for af_sarah.  Do not exceed 1.4 — quality degrades.
    kokoro_speed: float = field(default_factory=lambda: _env_float("TTS_KOKORO_SPEED", 1.2))
    # Native Kokoro output sample rate (do NOT change unless using a different model).
    sample_rate: int    = field(default_factory=lambda: _env_int("TTS_SAMPLE_RATE", 24000))

    # Native sample rate of the local output device.  Kokoro audio is resampled
    # to this rate before sd.play() to prevent CoreAudio crackling on macOS.
    # Auto-detected from sounddevice; override via TTS_DEVICE_SAMPLE_RATE.
    device_sample_rate: int = field(default_factory=lambda: _env_int(
        "TTS_DEVICE_SAMPLE_RATE", _get_device_sample_rate(),
    ))

    # ── Sentence splitting ────────────────────────────────────────────────────
    # Applied to accumulated LLM tokens to find sentence boundaries.
    # Only splits on terminal punctuation (.!?) followed by whitespace.
    # Commas, semicolons, colons, and em-dashes are intentionally excluded:
    # splitting at commas creates many short fragments that synthesize quickly
    # but produce audible gaps between them.  Longer sentence chunks benefit
    # fully from the pre-synthesis lookahead thread — sentence N+1 is already
    # synthesized before sentence N finishes playing.
    sentence_split_pattern: str = field(default_factory=lambda: _env_str(
        "TTS_SENTENCE_SPLIT",
        r'(?<=[.!?])\s+',
    ))

    # Flush the buffer immediately as a chunk when it exceeds this many
    # characters with no punctuation split yet.  Prevents run-on sentences
    # from stalling audio indefinitely.
    max_buffer_chars: int = field(default_factory=lambda: _env_int(
        "TTS_MAX_BUFFER_CHARS", 120,
    ))

    # Minimum number of words a chunk must have before it is synthesized
    # alone.  Chunks shorter than this are held and merged with the next.
    # 8 words ≈ ~50 chars — prevents very short clause fragments from
    # being synthesized as isolated calls.  Longer chunks mean fewer
    # synthesis calls total, and the pre-synthesis thread has more time
    # to complete lookahead synthesis before the current sentence ends.
    min_chunk_words: int = field(default_factory=lambda: _env_int(
        "TTS_MIN_CHUNK_WORDS", 8,
    ))

    # Minimum character length a non-sentence-final chunk must reach before
    # it is synthesized.  Chunks under this length are held and merged even
    # if they pass the word count guard.  Sentence-final punctuation (.!?)
    # bypasses this check and always flushes immediately.
    target_chunk_chars: int = field(default_factory=lambda: _env_int(
        "TTS_TARGET_CHUNK_CHARS", 80,
    ))

    # Seconds to wait before retrying a failed synthesis call.
    synthesis_retry_delay_s: float = field(default_factory=lambda: _env_float(
        "TTS_SYNTHESIS_RETRY_DELAY", 0.05,
    ))

    # ── Timing / latency ─────────────────────────────────────────────────────
    # Abort the turn if no LLM token arrives within this many seconds.
    llm_token_timeout_s: float = field(default_factory=lambda: _env_float(
        "LLM_TOKEN_TIMEOUT_S", 60.0,
    ))

    # How often (seconds) the sd.play polling loop checks for an interrupt.
    # Smaller = more responsive barge-in; larger = fewer syscalls.
    poll_interval_s: float = field(default_factory=lambda: _env_float(
        "TTS_POLL_INTERVAL_S", 0.02,
    ))

    # How often (seconds) the consumer loop polls the token queue when empty.
    consumer_poll_s: float = field(default_factory=lambda: _env_float(
        "TTS_CONSUMER_POLL_S", 0.05,
    ))

    # Timeout (seconds) on each queue.put() retry in the producer thread.
    producer_retry_s: float = field(default_factory=lambda: _env_float(
        "TTS_PRODUCER_RETRY_S", 0.1,
    ))

    # ── Queue sizing ──────────────────────────────────────────────────────────
    # Number of tokens the producer can buffer ahead of the TTS consumer.
    token_queue_size: int = field(default_factory=lambda: _env_int(
        "TTS_TOKEN_QUEUE_SIZE", 64,
    ))

    # Max retries to deliver the None sentinel when queue is full.
    # Total wait ≈ sentinel_retries × producer_retry_s (default: 100 × 0.1s = 10s).
    sentinel_retries: int = field(default_factory=lambda: _env_int(
        "TTS_SENTINEL_RETRIES", 100,
    ))

    # ── Logging ───────────────────────────────────────────────────────────────
    # Seconds between "waiting for first token..." heartbeat log lines.
    heartbeat_interval_s: float = field(default_factory=lambda: _env_float(
        "TTS_HEARTBEAT_INTERVAL_S", 5.0,
    ))

    # Max characters of a sentence shown in the "[tts] synth #N ..." log line.
    log_sentence_chars: int = field(default_factory=lambda: _env_int(
        "TTS_LOG_SENTENCE_CHARS", 55,
    ))

    # ── Remote audio delivery (WebSocket streaming) ───────────────────────────
    # Set TTS_REMOTE_AUDIO=true to stream synthesized audio to WS clients.
    # When false (default) audio is played locally only (no WS send).
    remote_audio: bool = field(default_factory=lambda:
        os.getenv("TTS_REMOTE_AUDIO", "false").lower() == "true"
    )

    # Audio format sent to WS clients: "pcm16" (raw signed-16-bit, no header)
    # or "opus" (compressed, requires opuslib).  Falls back to pcm16 if opus
    # is unavailable regardless of this setting.
    audio_format: str = field(default_factory=lambda: _env_str(
        "TTS_AUDIO_FORMAT", "pcm16",
    ))

    # Duration of each audio chunk sent over WS, in milliseconds.
    # Smaller → lower latency; larger → fewer messages.  Default: 20 ms.
    audio_chunk_ms: int = field(default_factory=lambda: _env_int(
        "TTS_AUDIO_CHUNK_MS", 20,
    ))

    # When true, ALSO play audio locally via sounddevice even when
    # remote_audio is enabled.  Useful for server-side monitoring.
    monitor_locally: bool = field(default_factory=lambda:
        os.getenv("TTS_MONITOR_LOCALLY", "false").lower() == "true"
    )


# Module-level singleton — imported by main_tts.py
TTS_CONFIG = TTSConfig()
