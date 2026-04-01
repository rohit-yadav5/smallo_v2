import io
import queue as _queue
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

# ── Abort flag ────────────────────────────────────────────────────────────────
# Instead of managing a raw PortAudio OutputStream we use sd.play() which is
# simpler, more reliable on macOS CoreAudio, and easily stopped via sd.stop().
# _abort_flag is set by abort_speaking() (from the VAD barge-in thread) and
# read by _synth_and_write() between chunk-plays.
_abort_flag = threading.Event()


def abort_speaking():
    """
    Immediately stop TTS playback.  Thread-safe.
    Called by the VAD loop on barge-in detection.

    IMPORTANT: this is the ONLY function that calls sd.stop().
    The main speak_stream thread must never call sd.stop() — two concurrent
    Pa_StopStream() calls on macOS CoreAudio trigger a SIGTRAP (signal -5).
    After abort_speaking() sets _abort_flag + stops the stream, the main
    thread's sd.get_stream().active polling loop exits naturally.
    """
    _abort_flag.set()
    try:
        # Guard: only stop if a stream is actually active.
        # Calling Pa_StopStream on an already-stopped stream can assert on macOS.
        if sd.get_stream().active:
            sd.stop()
    except Exception:
        pass  # RuntimeError if no current stream — safe to ignore


def _synthesize_to_buffer(sentence: str) -> tuple:
    """Synthesize a single sentence into a NumPy array. Returns (audio, sr)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file)
    buf.seek(0)
    audio, sr = sf.read(buf, dtype="float32")
    return audio, sr


def warmup():
    """Run a dummy synthesis so Piper ONNX JIT-compiles on startup."""
    _synthesize_to_buffer("Hi.")


def speak(text: str, interrupt_event=None):
    if not text.strip():
        return
    _abort_flag.clear()
    for sentence in [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]:
        if (interrupt_event and interrupt_event.is_set()) or _abort_flag.is_set():
            break
        audio, sr = _synthesize_to_buffer(sentence)
        sd.play(audio, samplerate=sr, blocking=False)
        while True:
            try:
                active = sd.get_stream().active
            except RuntimeError:
                break   # stream already gone
            if not active:
                break   # finished naturally
            if (interrupt_event and interrupt_event.is_set()) or _abort_flag.is_set():
                # abort_speaking() already called sd.stop() — don't call it again
                return
            time.sleep(0.02)


def speak_stream(token_gen, interrupt_event=None) -> tuple[str, dict]:
    """
    Stream tokens from the LLM and play audio sentence-by-sentence.

    Architecture
    ────────────
    _producer thread  — iterates the token generator (LLM HTTP stream), pushes
                        tokens to _tok_q.  Runs in a daemon thread so the main
                        thread can check interrupt every 50 ms even when the LLM
                        is slow to yield the next token (e.g. barge-in before
                        the first sentence is synthesised).

    Main thread       — polls _tok_q with a 50 ms timeout, accumulates tokens
                        into sentences, calls _synth_and_write per sentence.

    _synth_and_write  — synthesises with Piper, plays via sd.play() + 20 ms poll.
                        Uses sd.stop() for immediate barge-in cut (cleaner than
                        PortAudio OutputStream.abort() on macOS CoreAudio).

    Returns
    ───────
    (text_spoken, timing_dict)
    text_spoken     — full response text up to the interrupt (may be "")
    timing_dict     — first_word_secs, total_secs, token_count, sentence_count
    """
    _abort_flag.clear()

    _start        = time.perf_counter()
    _timing: dict = {}

    full_text     = ""
    buffer        = ""
    first_word    = True
    _token_count  = 0
    _sentence_num = 0

    LLM_TIMEOUT_S = 30.0    # abort if no token arrives within this many seconds
    _last_wait_log = -1     # heartbeat bucket index

    # ── Producer thread ───────────────────────────────────────────────────────
    _tok_q: _queue.Queue = _queue.Queue(maxsize=64)

    def _producer():
        _t0    = time.perf_counter()
        _first = True
        try:
            for token in token_gen:
                # Check abort FIRST — breaking the for-loop calls token_gen.close()
                # which propagates GeneratorExit through the generator chain and
                # cancels the underlying Ollama HTTP connection.  Without this, a
                # barge-in leaves a dangling Ollama request that blocks the next turn.
                if _abort_flag.is_set():
                    break
                if _first:
                    print(f"  [llm] ✓ first token  {time.perf_counter() - _t0:.3f}s", flush=True)
                    _first = False
                # Use a short timeout so the abort flag is checked regularly even
                # when the consumer (main loop) is slow or has already exited.
                while not _abort_flag.is_set():
                    try:
                        _tok_q.put(token, timeout=0.1)
                        break
                    except _queue.Full:
                        pass   # queue full — keep spinning until abort or space
        except Exception as e:
            print(f"  [llm] ✗ producer error: {e}", flush=True)
        finally:
            # Always deliver the sentinel so the main loop knows the stream is
            # done.  The main loop may be mid-synthesis (slow consumer), so the
            # queue can be full — retry briefly.  Give up only if abort fired
            # (main loop already exited and nobody reads the queue anymore).
            for _ in range(100):          # up to 10 s
                try:
                    _tok_q.put(None, timeout=0.1)
                    break
                except _queue.Full:
                    if _abort_flag.is_set():
                        break            # main loop gone; sentinel not needed

    threading.Thread(target=_producer, daemon=True, name="llm-producer").start()
    print(f"  [llm] producer started", flush=True)

    # ── Interrupt helper ──────────────────────────────────────────────────────

    def _stopped() -> bool:
        return (interrupt_event is not None and interrupt_event.is_set()) \
               or _abort_flag.is_set()

    # ── Per-sentence synthesis + playback ─────────────────────────────────────

    def _synth_and_write(sentence: str):
        nonlocal first_word, _sentence_num
        if _stopped():
            return

        _sentence_num += 1
        t0      = time.perf_counter()
        audio, sr = _synthesize_to_buffer(sentence)
        synth_s = time.perf_counter() - t0
        audio_s = len(audio) / sr
        print(
            f"  [tts] synth #{_sentence_num}  {synth_s:.2f}s → {audio_s:.2f}s audio"
            f"  '{sentence[:55]}{'…' if len(sentence) > 55 else ''}'",
            flush=True,
        )

        if _stopped():
            return

        if first_word:
            _timing["first_word_secs"] = time.perf_counter() - _start
            first_word = False

        # sd.play() + 20 ms poll — simple, reliable, immediately interruptible.
        # abort_speaking() is the SOLE caller of sd.stop(); we only break here
        # so that we never issue two concurrent Pa_StopStream() calls (SIGTRAP).
        sd.play(audio, samplerate=sr, blocking=False)
        while True:
            try:
                active = sd.get_stream().active
            except RuntimeError:
                break           # stream already cleaned up
            if not active:
                break           # finished naturally
            if _stopped():
                return          # abort_speaking() already stopped it — just exit
            time.sleep(0.02)

    # ── Token consumption loop ─────────────────────────────────────────────────
    try:
        while True:
            if _stopped():
                # abort_speaking() already called sd.stop() — just break cleanly
                break
            try:
                token = _tok_q.get(timeout=0.05)
            except _queue.Empty:
                if not full_text:
                    elapsed = time.perf_counter() - _start
                    bucket  = int(elapsed / 5)
                    if bucket > _last_wait_log:
                        _last_wait_log = bucket
                        print(f"  [llm] ◌ waiting for first token...  {elapsed:.0f}s", flush=True)
                    if elapsed > LLM_TIMEOUT_S:
                        print(
                            f"  [llm] ✗ no tokens in {LLM_TIMEOUT_S:.0f}s — "
                            f"Ollama may be hung (run: ollama ps)",
                            flush=True,
                        )
                        break
                continue

            if token is None:
                break   # generator exhausted or producer errored

            _token_count += 1
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
        # Nothing to do: either playback finished naturally, or abort_speaking()
        # already called sd.stop().  A second sd.stop() here would race with
        # abort_speaking() on CoreAudio and trigger a SIGTRAP (exit -5).
        pass

    _timing["total_secs"]     = time.perf_counter() - _start
    _timing["token_count"]    = _token_count
    _timing["sentence_count"] = _sentence_num
    _timing.setdefault("first_word_secs", _timing["total_secs"])

    if _token_count > 0:
        tok_per_s = _token_count / max(_timing["total_secs"], 0.001)
        print(
            f"  [llm] ■ {_token_count} tokens  {tok_per_s:.1f} tok/s"
            f"  first_word={_timing['first_word_secs']:.2f}s"
            f"  total={_timing['total_secs']:.2f}s",
            flush=True,
        )

    return full_text.strip(), _timing


if __name__ == "__main__":
    speak("Hello Rohit. This is Small O speaking. Using Piper text to speech.")
