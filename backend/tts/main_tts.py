import queue as _queue
import re
import threading
import time

import numpy as np
import sounddevice as sd

from tts.config import TTS_CONFIG
from tts.audio_utils import encode_pcm16, encode_opus, chunk_audio, resample_audio

# ── Kokoro singleton ──────────────────────────────────────────────────────────
# None until warmup() is called.
_kokoro = None   # kokoro_onnx.Kokoro instance

_SENTENCE_SPLIT = re.compile(TTS_CONFIG.sentence_split_pattern)

# ── Abort flag ────────────────────────────────────────────────────────────────
_abort_flag = threading.Event()

# ── Synthesis lock ────────────────────────────────────────────────────────────
# Kokoro's onnxruntime InferenceSession.run() is NOT thread-safe — 8-thread
# stress tests show non-deterministic output lengths, indicating shared mutable
# state in the session.  Both the main synthesis path and the pre-synthesis
# background thread serialize through this lock.
_synth_lock = threading.Lock()

# ── WebSocket audio sender hook ───────────────────────────────────────────────
# Registered by main.py so that synthesized audio can be streamed to browser
# clients without importing main.py (avoids circular imports).
#
# Signature: fn(audio: np.ndarray, sample_rate: int) -> None
#   Called once per synthesized sentence after synthesis, before sd.play().
#   Must be non-blocking (schedules async send via run_coroutine_threadsafe).
#
# Cleared back to None when the connection pool is empty.
_ws_audio_sender = None   # Callable[[np.ndarray, int], None] | None


def register_ws_audio_sender(fn) -> None:
    """Register (or clear) the WebSocket audio send callback."""
    global _ws_audio_sender
    _ws_audio_sender = fn


def _send_audio_to_ws(audio: np.ndarray, sr: int) -> None:
    """
    Send synthesized audio to all connected WebSocket clients.

    Chunks audio → encodes each chunk → sends metadata frame (JSON) then
    each chunk as raw bytes.  All sends are non-blocking (fire-and-forget
    via the registered sender callback).

    No-op when remote_audio is disabled or no sender is registered.
    """
    if not TTS_CONFIG.remote_audio or _ws_audio_sender is None:
        return

    encode_fn = encode_opus if TTS_CONFIG.audio_format == "opus" else encode_pcm16
    chunks    = chunk_audio(audio, sr, TTS_CONFIG.audio_chunk_ms)
    if not chunks:
        return

    # Metadata frame: tells the client how many chunks to expect + format details
    _ws_audio_sender({
        "event": "AUDIO_CHUNK",
        "data": {
            "format":      TTS_CONFIG.audio_format,
            "sample_rate": sr,
            "chunk_count": len(chunks),
            "chunk_ms":    TTS_CONFIG.audio_chunk_ms,
        },
    }, None)

    # Binary frames: one per chunk
    for chunk in chunks:
        _ws_audio_sender(None, encode_fn(chunk, sr))


def _send_audio_abort_to_ws() -> None:
    """Signal clients to stop buffered audio immediately (barge-in)."""
    if not TTS_CONFIG.remote_audio or _ws_audio_sender is None:
        return
    _ws_audio_sender({"event": "AUDIO_ABORT", "data": {}}, None)


def _send_audio_done_to_ws() -> None:
    """Signal clients that the current utterance audio stream is complete."""
    if not TTS_CONFIG.remote_audio or _ws_audio_sender is None:
        return
    _ws_audio_sender({"event": "AUDIO_DONE", "data": {}}, None)


def _safe_stop() -> None:
    """
    The ONE place sd.stop() is ever called in this module.

    macOS CoreAudio SIGTRAP risk
    ────────────────────────────
    Two concurrent Pa_StopStream() calls on macOS CoreAudio trigger a
    SIGTRAP (signal -5, process killed).  This happens if both the barge-in
    thread and the playback polling loop call sd.stop() simultaneously.

    Rule: ONLY abort_speaking() calls _safe_stop().  The playback polling
    loop in speak() / _synth_and_write() must NEVER call sd.stop() — it
    checks _abort_flag and returns, trusting that abort_speaking() has
    already (or will soon) stop the stream.

    Guard: check sd.get_stream().active before stopping to avoid calling
    Pa_StopStream on an already-stopped stream (also causes an assert on
    macOS).
    """
    try:
        if sd.get_stream().active:
            sd.stop()
    except Exception:
        pass  # RuntimeError if no current stream — safe to ignore


def abort_speaking():
    """
    Immediately stop TTS playback.  Thread-safe.
    Called by the VAD loop on barge-in detection.
    """
    _abort_flag.set()
    _safe_stop()
    _send_audio_abort_to_ws()


# ── Low-level synthesis ───────────────────────────────────────────────────────

def _synthesize_to_buffer(text: str) -> tuple[np.ndarray, int]:
    """
    Synthesize text → (float32 audio array, sample_rate).

    Acquires _synth_lock because Kokoro's InferenceSession is not thread-safe.
    Called by both the main thread and the pre-synthesis background thread.
    """
    with _synth_lock:
        audio, sr = _kokoro.create(
            text,
            voice=TTS_CONFIG.kokoro_voice,
            speed=TTS_CONFIG.kokoro_speed,
            lang="en-us",
        )
        return audio.astype(np.float32), sr


# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup() -> None:
    """
    Load Kokoro and force ONNX JIT compile with a short synthesis so the
    first real utterance has no cold-start penalty.
    """
    global _kokoro
    import os
    from kokoro_onnx import Kokoro
    model_path  = os.path.normpath(TTS_CONFIG.kokoro_model_path)
    voices_path = os.path.normpath(TTS_CONFIG.kokoro_voices_path)
    _kokoro = Kokoro(model_path, voices_path)
    _synthesize_to_buffer("Hi.")
    print(
        f"  [tts] Kokoro ready  voice={TTS_CONFIG.kokoro_voice}  "
        f"speed={TTS_CONFIG.kokoro_speed}  sr={TTS_CONFIG.sample_rate}",
        flush=True,
    )
    if TTS_CONFIG.sample_rate != TTS_CONFIG.device_sample_rate:
        print(
            f"  [tts] resampling active: "
            f"Kokoro {TTS_CONFIG.sample_rate} Hz → device {TTS_CONFIG.device_sample_rate} Hz",
            flush=True,
        )
    else:
        print(
            f"  [tts] no resampling needed — "
            f"device matches Kokoro at {TTS_CONFIG.sample_rate} Hz",
            flush=True,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def speak(text: str, interrupt_event=None):
    # speak() is used only by warmup() — production uses speak_stream()
    if not text.strip():
        return
    _abort_flag.clear()
    for sentence in [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]:
        if (interrupt_event and interrupt_event.is_set()) or _abort_flag.is_set():
            break
        audio, sr = _synthesize_to_buffer(sentence)
        audio_out = resample_audio(audio, sr, TTS_CONFIG.device_sample_rate)
        sd.play(audio_out, samplerate=TTS_CONFIG.device_sample_rate, blocking=False)
        while True:
            try:
                active = sd.get_stream().active
            except RuntimeError:
                break
            if not active:
                break
            if (interrupt_event and interrupt_event.is_set()) or _abort_flag.is_set():
                return
            time.sleep(TTS_CONFIG.poll_interval_s)


def speak_stream(token_gen, interrupt_event=None) -> tuple[str, dict]:
    """
    Stream tokens from the LLM and play audio sentence-by-sentence.

    Architecture
    ────────────
    _producer thread  — iterates the token generator (LLM HTTP stream), pushes
                        tokens to _tok_q.  Runs in a daemon thread so the main
                        thread can check interrupt every consumer_poll_s even when
                        the LLM is slow to yield the next token.

    Main thread       — polls _tok_q, accumulates tokens into a buffer, extracts
                        complete chunks via _SENTENCE_SPLIT and max_buffer_chars,
                        enforces min_chunk_words merging, calls _synth_and_write
                        per ready chunk.

    _synth_and_write  — synthesises with the active engine (with retry on failure),
                        plays via sd.play() + poll_interval_s poll, and kicks off
                        a background pre-synthesis thread for the next queued chunk
                        while current audio is playing.

    Pre-synthesis     — after sd.play() starts, if the next chunk is already known,
                        a daemon thread synthesizes it in parallel under _synth_lock.
                        When playback finishes, the pre-synthesized result is used
                        directly (zero synthesis wait).  Falls back to synchronous
                        synthesis if the thread isn't done yet.

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

    _last_wait_log = -1

    # ── Pre-synthesis slot ────────────────────────────────────────────────────
    _presyn: dict = {}

    def _run_presyn(sentence: str, slot: dict) -> None:
        try:
            slot["result"] = _synthesize_to_buffer(sentence)
        except Exception:
            slot["result"] = None
        finally:
            slot["done"].set()

    # ── Producer thread ───────────────────────────────────────────────────────
    _tok_q: _queue.Queue = _queue.Queue(maxsize=TTS_CONFIG.token_queue_size)

    def _producer():
        _t0    = time.perf_counter()
        _first = True
        try:
            for token in token_gen:
                if _abort_flag.is_set():
                    break
                if _first:
                    _timing["first_token_secs"] = time.perf_counter() - _start
                    print(f"  [llm] ✓ first token  {time.perf_counter() - _t0:.3f}s", flush=True)
                    _first = False
                while not _abort_flag.is_set():
                    try:
                        _tok_q.put(token, timeout=TTS_CONFIG.producer_retry_s)
                        break
                    except _queue.Full:
                        pass
        except Exception as e:
            print(f"  [llm] ✗ producer error: {e}", flush=True)
        finally:
            for _ in range(TTS_CONFIG.sentinel_retries):
                try:
                    _tok_q.put(None, timeout=TTS_CONFIG.producer_retry_s)
                    break
                except _queue.Full:
                    if _abort_flag.is_set():
                        break

    threading.Thread(target=_producer, daemon=True, name="llm-producer").start()
    print(f"  [llm] producer started", flush=True)

    # ── Interrupt helper ──────────────────────────────────────────────────────

    def _stopped() -> bool:
        return (interrupt_event is not None and interrupt_event.is_set()) \
               or _abort_flag.is_set()

    # ── Synthesis with retry ──────────────────────────────────────────────────

    def _synthesize_with_retry(sentence: str) -> tuple | None:
        try:
            return _synthesize_to_buffer(sentence)
        except Exception as e1:
            print(f"  [tts] ✗ synthesis error (will retry): {e1}", flush=True)
            time.sleep(TTS_CONFIG.synthesis_retry_delay_s)
            try:
                return _synthesize_to_buffer(sentence)
            except Exception as e2:
                n = TTS_CONFIG.log_sentence_chars
                print(
                    f"  [tts] ✗ synthesis failed, skipping: {sentence[:n]!r}  ({e2})",
                    flush=True,
                )
                return None

    # ── Per-sentence synthesis + playback ─────────────────────────────────────

    def _synth_and_write(sentence: str, next_sentence: str | None = None):
        nonlocal first_word, _sentence_num
        if _stopped():
            return

        _sentence_num += 1

        # Check if pre-synthesized audio is ready from previous iteration
        audio, sr = None, None
        if _presyn.get("text") == sentence:
            _presyn["done"].wait()
            audio, sr = _presyn.get("result") or (None, None)
            _presyn.clear()
            synth_label = "pre-synth" if audio is not None else None

        if audio is None:
            t0     = time.perf_counter()
            result = _synthesize_with_retry(sentence)
            if result is None:
                return
            audio, sr  = result
            synth_s    = time.perf_counter() - t0
            synth_label = f"{synth_s:.2f}s"

        audio_s = len(audio) / sr
        n       = TTS_CONFIG.log_sentence_chars
        print(
            f"  [tts] synth #{_sentence_num}  {synth_label} → {audio_s:.2f}s audio"
            f"  '{sentence[:n]}{'…' if len(sentence) > n else ''}'",
            flush=True,
        )

        if _stopped():
            return

        if first_word:
            _timing["first_word_secs"] = time.perf_counter() - _start
            first_word = False

        # ── Remote audio delivery ─────────────────────────────────────────
        _send_audio_to_ws(audio, sr)

        # ── Local playback ────────────────────────────────────────────────
        if not TTS_CONFIG.remote_audio or TTS_CONFIG.monitor_locally:
            audio_out = resample_audio(audio, sr, TTS_CONFIG.device_sample_rate)
            sd.play(audio_out, samplerate=TTS_CONFIG.device_sample_rate, blocking=False)

        # Start pre-synthesis of next sentence while current audio plays
        if next_sentence and not _stopped():
            slot: dict = {"text": next_sentence, "result": None,
                          "done": threading.Event()}
            _presyn.clear()
            _presyn.update(slot)
            threading.Thread(
                target=_run_presyn,
                args=(next_sentence, slot),
                daemon=True,
                name="tts-presyn",
            ).start()

        # Polling loop: wait for local playback to finish.
        # Skipped when remote_audio is on and monitor_locally is off
        # (no sd.play() was called so there is no stream to poll).
        if not TTS_CONFIG.remote_audio or TTS_CONFIG.monitor_locally:
            while True:
                try:
                    active = sd.get_stream().active
                except RuntimeError:
                    break
                if not active:
                    break
                if _stopped():
                    return
                time.sleep(TTS_CONFIG.poll_interval_s)

    # ── Chunking helpers ──────────────────────────────────────────────────────

    _held: str = ""

    def _maybe_emit(chunk: str) -> str | None:
        nonlocal _held
        # Combine any previously held fragment with the incoming chunk.
        combined = (_held + " " + chunk).strip() if _held else chunk
        words    = combined.split()

        # Sentence-final punctuation (.!?) always flushes immediately —
        # these are natural pause points where the listener expects audio.
        # Comma/semicolon splits are NOT sentence-final and must pass both
        # the word-count and char-length guards before being synthesized.
        if combined.rstrip().endswith(('.', '!', '?')):
            _held = ""
            return combined

        # Hold if the combined chunk is still too short by either measure.
        if (len(words) < TTS_CONFIG.min_chunk_words
                or len(combined) < TTS_CONFIG.target_chunk_chars):
            _held = combined
            return None

        _held = ""
        return combined

    def _flush_held() -> str | None:
        nonlocal _held
        if _held:
            out   = _held
            _held = ""
            return out
        return None

    _ready: list[str] = []

    def _enqueue_chunk(raw: str) -> None:
        out = _maybe_emit(raw.strip())
        if out:
            _ready.append(out)

    # ── Token consumption loop ─────────────────────────────────────────────────
    try:
        while True:
            if _stopped():
                break
            try:
                token = _tok_q.get(timeout=TTS_CONFIG.consumer_poll_s)
            except _queue.Empty:
                if not full_text:
                    elapsed = time.perf_counter() - _start
                    bucket  = int(elapsed / TTS_CONFIG.heartbeat_interval_s)
                    if bucket > _last_wait_log:
                        _last_wait_log = bucket
                        print(f"  [llm] ◌ waiting for first token...  {elapsed:.0f}s", flush=True)
                    if elapsed > TTS_CONFIG.llm_token_timeout_s:
                        print(
                            f"  [llm] ✗ no tokens in {TTS_CONFIG.llm_token_timeout_s:.0f}s — "
                            f"Ollama may be hung (run: ollama ps)",
                            flush=True,
                        )
                        break
                continue

            if token is None:
                break

            _token_count += 1
            buffer    += token
            full_text += token

            parts = _SENTENCE_SPLIT.split(buffer)
            if len(parts) > 1:
                for part in parts[:-1]:
                    if part.strip():
                        _enqueue_chunk(part)
                buffer = parts[-1]
            elif len(buffer) >= TTS_CONFIG.max_buffer_chars:
                _enqueue_chunk(buffer)
                buffer = ""

            while _ready and not _stopped():
                current = _ready.pop(0)
                nxt     = _ready[0] if _ready else None
                _synth_and_write(current, next_sentence=nxt)

        # Drain
        if buffer.strip():
            _enqueue_chunk(buffer.strip())
        held = _flush_held()
        if held:
            _ready.append(held)

        while _ready and not _stopped():
            current = _ready.pop(0)
            nxt     = _ready[0] if _ready else None
            _synth_and_write(current, next_sentence=nxt)

    finally:
        pass   # abort_speaking() owns sd.stop() — never call it here

    # Signal clients that all audio for this utterance has been sent.
    # Skipped on barge-in (_abort_flag set) because abort_speaking() already
    # sent AUDIO_ABORT and the client has already flushed its queue.
    if not _abort_flag.is_set():
        _send_audio_done_to_ws()

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
    warmup()
    speak("Hello Rohit. This is Small O speaking. Using Kokoro text to speech.")
