import io
import re
import queue
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


def speak(text: str):
    if not text.strip():
        return

    sentences = _split_sentences(text)

    # Fallback: if no sentence boundaries found, treat as one chunk
    if not sentences:
        sentences = [text.strip()]

    # Producer: synthesize sentences in background and put into queue
    audio_queue: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        for sentence in sentences:
            audio, sr = _synthesize_to_buffer(sentence)
            audio_queue.put((audio, sr))
        audio_queue.put(None)  # sentinel

    synth_thread = threading.Thread(target=producer, daemon=True)
    synth_thread.start()

    # Consumer: play each sentence as soon as it's ready
    while True:
        item = audio_queue.get()
        if item is None:
            break
        audio, sr = item
        sd.play(audio, samplerate=sr)
        sd.wait()


def speak_stream(token_gen) -> tuple[str, dict]:
    """
    Consume a token generator from the LLM and play audio sentence-by-sentence.

    Uses a persistent sd.OutputStream so the audio device stays open across
    sentences — no gaps, no re-open overhead. stream.write() blocks when the
    hardware ring buffer is full, which naturally overlaps synthesis of sentence
    N+1 with playback of sentence N. stream.stop() (Pa_StopStream) is guaranteed
    to wait until every buffered sample has been played before returning.

    Returns:
        text    — the full text spoken
        timing  — dict with keys:
                    first_word_secs  — time from call start to first audio ready
                    total_secs       — total wall time including all playback
    """
    _start = time.perf_counter()
    _timing: dict = {}

    full_text = ""
    buffer = ""
    stream = None
    first = True

    def _synth_and_write(sentence: str):
        nonlocal stream, first
        print(f"    [tts] speaking: {sentence!r:.60}")
        audio, sr = _synthesize_to_buffer(sentence)

        if stream is None:
            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            stream.start()

        if first:
            _timing["first_word_secs"] = time.perf_counter() - _start
            first = False

        stream.write(audio)  # blocks when ring buffer full → synthesis overlaps playback

    for token in token_gen:
        buffer += token
        full_text += token

        parts = _SENTENCE_SPLIT.split(buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    _synth_and_write(sentence)
            buffer = parts[-1]

    if buffer.strip():
        _synth_and_write(buffer.strip())

    if stream is not None:
        stream.stop()   # Pa_StopStream: blocks until all buffered audio has played
        stream.close()

    _timing["total_secs"] = time.perf_counter() - _start
    _timing.setdefault("first_word_secs", _timing["total_secs"])
    return full_text.strip(), _timing


if __name__ == "__main__":
    speak("Hello Rohit. This is Small O speaking. Using Piper text to speech. The streaming should feel much faster now.")
