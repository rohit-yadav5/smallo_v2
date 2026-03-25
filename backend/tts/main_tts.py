import io
import re
import threading
import time
import wave

import sounddevice as sd
import soundfile as sf
from piper.voice import PiperVoice
import os

VOICE_PATH = os.path.expanduser(
    "~/piper-voices/en_US-amy-medium.onnx"
)

voice = PiperVoice.load(VOICE_PATH)
voice.config.length_scale = 0.7  # <1.0 = faster, >1.0 = slower

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# Module-level reference to the active OutputStream so abort_speaking()
# can stop it from any thread (VAD loop, pipeline, etc.)
_stream_lock   = threading.Lock()
_active_stream: sd.OutputStream | None = None


def _set_active_stream(s: sd.OutputStream | None):
    global _active_stream
    with _stream_lock:
        _active_stream = s


def abort_speaking():
    """
    Immediately cut TTS audio playback. Thread-safe.
    Called by the VAD loop on barge-in detection.
    """
    with _stream_lock:
        s = _active_stream
    if s is not None:
        try:
            s.abort()   # Pa_AbortStream: drops buffered audio immediately
        except Exception:
            pass


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _synthesize_to_buffer(sentence: str) -> tuple:
    """Synthesize a single sentence into an in-memory buffer. Returns (audio, sr)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file)
    buf.seek(0)
    audio, sr = sf.read(buf, dtype="float32")
    return audio, sr


def warmup():
    """Run a dummy synthesis so Piper ONNX JIT-compiles on startup, not on first response."""
    _synthesize_to_buffer("Hi.")


def speak(text: str, interrupt_event=None):
    if not text.strip():
        return

    sentences = _split_sentences(text) or [text.strip()]

    import queue as _queue
    audio_queue: _queue.Queue = _queue.Queue(maxsize=2)

    def producer():
        for sentence in sentences:
            if interrupt_event and interrupt_event.is_set():
                break
            audio, sr = _synthesize_to_buffer(sentence)
            audio_queue.put((audio, sr))
        audio_queue.put(None)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        item = audio_queue.get()
        if item is None:
            break
        if interrupt_event and interrupt_event.is_set():
            break
        audio, sr = item
        sd.play(audio, samplerate=sr)
        sd.wait()


def speak_stream(token_gen, interrupt_event=None) -> tuple[str, dict]:
    """
    Consume a token generator from the LLM and play audio sentence-by-sentence.

    interrupt_event — threading.Event that, when set, stops playback immediately.
    Called by the VAD loop when user speaks during TTS (barge-in).

    Returns:
        text    — the full text spoken (may be partial if interrupted)
        timing  — dict with first_word_secs and total_secs
    """
    _start = time.perf_counter()
    _timing: dict = {}

    full_text = ""
    buffer    = ""
    stream: sd.OutputStream | None = None
    first = True

    # Write chunk size for interrupt granularity: ~93ms @ 22050 Hz
    WRITE_CHUNK = 2048

    def _stopped() -> bool:
        return interrupt_event is not None and interrupt_event.is_set()

    def _synth_and_write(sentence: str):
        nonlocal stream, first
        if _stopped():
            return

        print(f"    [tts] speaking: {sentence!r:.60}")
        audio, sr = _synthesize_to_buffer(sentence)

        if _stopped():
            return

        if stream is None:
            s = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            s.start()
            _set_active_stream(s)
            stream = s

        if first:
            _timing["first_word_secs"] = time.perf_counter() - _start
            first = False

        # Write in small chunks so we can react to interrupt quickly
        for i in range(0, len(audio), WRITE_CHUNK):
            if _stopped():
                return
            stream.write(audio[i : i + WRITE_CHUNK])

    # ── Token consumption loop ─────────────────────────────────────────────
    try:
        for token in token_gen:
            if _stopped():
                break
            buffer    += token
            full_text += token

            parts = _SENTENCE_SPLIT.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        _synth_and_write(sentence)
                buffer = parts[-1]

        if buffer.strip() and not _stopped():
            _synth_and_write(buffer.strip())

    finally:
        if stream is not None:
            _set_active_stream(None)
            try:
                if _stopped():
                    stream.abort()   # immediate cut — barge-in
                else:
                    stream.stop()    # drain remaining samples cleanly
                stream.close()
            except Exception:
                pass

    _timing["total_secs"] = time.perf_counter() - _start
    _timing.setdefault("first_word_secs", _timing["total_secs"])
    return full_text.strip(), _timing


if __name__ == "__main__":
    speak("Hello Rohit. This is Small O speaking. Using Piper text to speech. The streaming should feel much faster now.")
