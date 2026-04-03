"""
stt/streaming.py — StreamingTranscriber: live word-by-word transcription.

Architecture
────────────
Traditional STT waits until the full utterance is finished, then sends all
audio to Whisper at once.  StreamingTranscriber runs Whisper incrementally
while the user is still speaking:

  1. VAD feeds every 16 ms speech chunk via feed().
  2. Every 500 ms of new accumulated audio, Whisper is run in a background
     thread on the full audio seen so far.
  3. Local agreement: words that appear at the same index across two
     consecutive Whisper runs are "confirmed" and emitted immediately.
  4. Unconfirmed trailing words are emitted as "hypothesis" (faded in UI).
  5. At the first silence frame, start_finalize() queues a final Whisper
     call in the background.
  6. When the VAD seals the utterance, finalize() returns the best result:
     - the background snapshot result if the audio hasn't grown too much, or
     - a fresh Whisper call on the complete utterance for maximum accuracy.

Result: words appear on screen while the user is still speaking, similar to
Google Live Transcribe, then the final result corrects any partial errors.
"""
import threading

import numpy as np
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable


class StreamingTranscriber:
    """Live streaming STT with local-agreement word confirmation.

    Args:
        transcribe_fn:    Function with signature (np.ndarray) -> (str, float).
                          Must be thread-safe (handled by stt_lock in engine.py).
        on_partial:       Callback called whenever new confirmed/hypothesis text
                          is available.  Signature: (confirmed: str, hypothesis: str).
                          Called from a background thread — must be non-blocking.
        chunk_interval_s: How often to run Whisper during speech (seconds of new
                          audio between runs).  Default 0.5 s is a good balance
                          between responsiveness and CPU load.
    """

    def __init__(
        self,
        transcribe_fn:          Callable[[np.ndarray], tuple[str, float]],
        on_partial:             Callable[[str, str], None],
        chunk_interval_s:       float    = 0.3,          # was 0.5 — tighter = more responsive
        transcribe_partial_fn:  Callable | None = None,  # fast beam_size=1 path for live display
    ):
        self._transcribe         = transcribe_fn
        self._transcribe_partial = transcribe_partial_fn
        self._on_partial         = on_partial
        self._interval_samples   = int(chunk_interval_s * 16_000)

        # Audio accumulation
        self._chunks:            list[np.ndarray] = []
        self._samples_since_run: int              = 0

        # Agreement state
        self._confirmed_words: list[str]        = []
        self._prev_words:      list[str]        = []
        self._prev_times:      dict[int, float] = {}   # word index → start_secs from prev run
        self._last_emitted:    str              = ""   # guards against duplicate emits

        # Thread pool — single worker so Whisper calls never overlap
        self._pool:    ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stt-live"
        )
        self._partial: Future | None = None   # in-flight partial transcription
        self._final:   Future | None = None   # in-flight finalize transcription

        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def feed(self, chunk: np.ndarray) -> None:
        """Receive a 16 ms speech chunk from the VAD loop.  Thread-safe.

        Accumulates audio and fires a background Whisper call every
        chunk_interval_s of new audio.  If the previous Whisper call is still
        running, the new call is skipped (the next interval will catch up with
        more audio) — this prevents queue pile-up when Whisper is slower than
        the audio stream.
        """
        with self._lock:
            self._chunks.append(chunk)
            self._samples_since_run += len(chunk)

            if self._samples_since_run < self._interval_samples:
                return   # not enough new audio yet
            self._samples_since_run = 0

            # Skip if previous partial still running
            if self._partial is not None and not self._partial.done():
                return

            audio = np.concatenate(self._chunks)
            self._partial = self._pool.submit(self._run_partial, audio.copy())

    def start_finalize(self, snapshot: np.ndarray) -> None:
        """Called at the first silence frame — start final Whisper in background.

        By the time silence_ms elapses confirming the utterance is over, the
        transcription is likely already done.  finalize() will use this result
        if the audio hasn't grown significantly since the snapshot was taken.
        """
        with self._lock:
            self._final = self._pool.submit(self._transcribe, snapshot.copy())

    def finalize(self, full_audio: np.ndarray) -> tuple[str, float]:
        """Return the final transcription for the completed utterance.

        Decision logic:
          1. Wait for any in-progress partial (up to 2.5 s).
          2. If the background snapshot result is ready AND the full audio
             is not significantly longer than the snapshot (≤ 1.5×), use it —
             saves the full inference time.
          3. Otherwise, run a fresh Whisper call on the complete utterance.
        """
        # Let any in-flight partial drain first — it shares the pool so waiting
        # here also prevents it from racing with the finalize result.
        if self._partial is not None:
            try:
                self._partial.result(timeout=2.5)
            except Exception:
                pass

        with self._lock:
            ff       = self._final
            snap_len = sum(len(c) for c in self._chunks)

        if ff is not None:
            try:
                text, secs = ff.result(timeout=2.5)
                if text.strip() and len(full_audio) <= snap_len * 1.5:
                    return text, secs
            except Exception:
                pass

        # Fresh transcription on the complete utterance
        return self._transcribe(full_audio)

    def reset(self) -> None:
        """Clear all state.  Call when VAD resets between turns."""
        with self._lock:
            self._chunks.clear()
            self._confirmed_words.clear()
            self._prev_words.clear()
            self._prev_times.clear()
            self._samples_since_run = 0
            self._last_emitted      = ""
            self._partial           = None
            self._final             = None

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_time_map(
        words_list: list[tuple[str, float, float]],
        curr_words: list[str],
    ) -> dict[int, float]:
        """Map curr_words index → start_secs from faster-whisper word list.

        faster-whisper word tokens carry leading spaces and attached punctuation
        (e.g. ' hello', ' world.').  curr_words comes from text.split().
        Greedy scan: advance through words_list matching cleaned strings.
        Indices with no match are absent — caller falls back to text-only agreement.
        """
        if not words_list:
            return {}
        time_map: dict[int, float] = {}
        j = 0
        for i, cw in enumerate(curr_words):
            cw_clean = cw.lower().strip(".,!?;:'\"")
            while j < len(words_list):
                fw_text, fw_start, _ = words_list[j]
                fw_clean = fw_text.strip().lower().strip(".,!?;:'\"")
                j += 1
                if fw_clean == cw_clean:
                    time_map[i] = fw_start
                    break
        return time_map

    def _run_partial(self, audio: np.ndarray) -> None:
        """Runs in the pool thread.  Calls Whisper, applies upgraded agreement,
        fires on_partial if confirmed or hypothesis text changed.

        Agreement upgrade: word i is confirmed only when:
          1. text matches prev_words[i] (case/punct-insensitive), AND
          2. if word timestamps are available for both runs:
             |curr_time[i] - prev_time[i]| < 100ms
        The timestamp check prevents jitter when Whisper rewrites punctuation
        or capitalisation — the word position in the audio timeline is stable
        even when the surface form changes slightly.
        """
        if self._transcribe_partial is not None:
            text, words_list, _ = self._transcribe_partial(audio)
        else:
            text, _ = self._transcribe(audio)
            words_list = []

        if not text.strip():
            return

        curr_words = text.split()
        curr_times = self._build_time_map(words_list, curr_words)

        with self._lock:
            agree = 0
            for i in range(len(curr_words)):
                if i >= len(self._prev_words):
                    break
                text_match = (
                    self._prev_words[i].lower().strip(".,!?") ==
                    curr_words[i].lower().strip(".,!?")
                )
                if not text_match:
                    break
                # Timestamp stability: require ≤100ms drift when both runs have data
                if i in curr_times and i in self._prev_times:
                    if abs(curr_times[i] - self._prev_times[i]) >= 0.10:
                        break
                agree += 1

            self._prev_words = curr_words
            self._prev_times = curr_times

            # Only advance confirmed count — never retract already-confirmed words.
            if agree > len(self._confirmed_words):
                self._confirmed_words = curr_words[:agree]

            confirmed  = " ".join(self._confirmed_words)
            hypothesis = " ".join(curr_words[len(self._confirmed_words):])
            emit_key   = f"{confirmed}|{hypothesis}"

        # Avoid emitting the same text twice
        if emit_key != self._last_emitted and (confirmed or hypothesis):
            self._last_emitted = emit_key
            self._on_partial(confirmed, hypothesis)
