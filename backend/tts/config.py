"""tts/config.py — TTS configuration.

Every value has a hardcoded default that matches the original behaviour.
All values are overridable via environment variable — useful for testing
a different voice, adjusting speed, or tuning latency without touching code.

Usage
─────
    from tts.config import TTS_CONFIG

    voice = PiperVoice.load(TTS_CONFIG.voice_path)
    voice.config.length_scale = TTS_CONFIG.length_scale

Environment variables
─────────────────────
    TTS_ENGINE              Which engine to use: "kokoro" (default) or "piper"

    --- Kokoro (default engine) ---
    TTS_KOKORO_MODEL_PATH   Path to kokoro-v1.0.onnx (default: kokoro-models/ in project root)
    TTS_KOKORO_VOICES_PATH  Path to voices-v1.0.bin  (default: kokoro-models/ in project root)
    TTS_KOKORO_VOICE        Voice name, e.g. af_sarah, bf_emma (default: af_sarah)
    TTS_KOKORO_SPEED        Speech speed multiplier; 1.0 = normal (default: 1.1)
    TTS_SAMPLE_RATE         Native output sample rate in Hz (default: 24000 for Kokoro)

    --- Piper (fallback engine, set TTS_ENGINE=piper) ---
    PIPER_VOICE_PATH        Path to the .onnx voice model file
    PIPER_LENGTH_SCALE      Speech speed  (<1.0 = faster, >1.0 = slower)
    PIPER_NOISE_SCALE       Phoneme duration variability (Piper default 0.667)
    PIPER_NOISE_W           Phoneme width variability (Piper default 0.8)

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
"""
import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class TTSConfig:
    # ── Engine selection ──────────────────────────────────────────────────────
    # "kokoro" (default) or "piper" for fallback to the old Piper engine.
    engine: str = field(default_factory=lambda: _env_str("TTS_ENGINE", "kokoro"))

    # ── Kokoro settings ───────────────────────────────────────────────────────
    # Paths default to the kokoro-models/ directory at the project root.
    # The _PROJ_ROOT calculation assumes this file lives at backend/tts/config.py.
    kokoro_model_path: str = field(default_factory=lambda: _env_str(
        "TTS_KOKORO_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "kokoro-models", "kokoro-v1.0.onnx"),
    ))
    kokoro_voices_path: str = field(default_factory=lambda: _env_str(
        "TTS_KOKORO_VOICES_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "kokoro-models", "voices-v1.0.bin"),
    ))
    # Voice name.  af_sarah chosen as default: American female, RTF≈0.26 (2× faster
    # than af_heart RTF≈0.52) based on benchmarking against af_heart and bf_emma.
    kokoro_voice: str   = field(default_factory=lambda: _env_str("TTS_KOKORO_VOICE", "af_sarah"))
    # Speed multiplier passed to Kokoro.create(speed=...).  1.1 = slightly faster.
    kokoro_speed: float = field(default_factory=lambda: _env_float("TTS_KOKORO_SPEED", 1.1))
    # Native Kokoro output sample rate (do NOT change unless using a different model).
    sample_rate: int    = field(default_factory=lambda: _env_int("TTS_SAMPLE_RATE", 24000))

    # ── Piper settings (used only when TTS_ENGINE=piper) ─────────────────────
    voice_path: str = field(default_factory=lambda: _env_str(
        "PIPER_VOICE_PATH",
        os.path.expanduser("~/piper-voices/en_US-amy-medium.onnx"),
    ))
    # length_scale < 1.0 = faster speech; > 1.0 = slower speech
    length_scale: float = field(default_factory=lambda: _env_float("PIPER_LENGTH_SCALE", 0.7))
    # noise_scale controls phoneme duration variance (Piper default 0.667)
    noise_scale: float  = field(default_factory=lambda: _env_float("PIPER_NOISE_SCALE", 0.667))
    # noise_w controls phoneme width variance (Piper default 0.8)
    noise_w: float      = field(default_factory=lambda: _env_float("PIPER_NOISE_W", 0.8))

    # ── Sentence splitting ────────────────────────────────────────────────────
    # Applied to accumulated LLM tokens to find sentence boundaries.
    # Splits after .!? and also after ,;:— so comma-heavy sentences don't
    # hold the buffer until the final period (2-3s of silence avoided).
    sentence_split_pattern: str = field(default_factory=lambda: _env_str(
        "TTS_SENTENCE_SPLIT",
        r'(?<=[.!?,;:\u2014])\s+',
    ))

    # Flush the buffer immediately as a chunk when it exceeds this many
    # characters with no punctuation split yet.  Prevents run-on sentences
    # from stalling audio indefinitely.
    max_buffer_chars: int = field(default_factory=lambda: _env_int(
        "TTS_MAX_BUFFER_CHARS", 120,
    ))

    # Minimum number of words a chunk must have before it is synthesized
    # alone.  Chunks shorter than this are held and merged with the next
    # chunk to avoid wasting Piper startup overhead on tiny fragments.
    min_chunk_words: int = field(default_factory=lambda: _env_int(
        "TTS_MIN_CHUNK_WORDS", 3,
    ))

    # Seconds to wait before retrying a failed Piper synthesis call.
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
