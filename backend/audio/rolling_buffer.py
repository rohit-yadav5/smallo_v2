"""
audio/rolling_buffer.py — Always-on 60-second circular audio buffer.

Architecture
────────────
The microphone NEVER stops recording.  Every audio frame goes here first,
before any VAD or STT processing.  This means any time window in the past
60 seconds can be retrieved by session-elapsed timestamp.

  Mic → _audio_ingestion_loop → RollingAudioBuffer.write()
                                         │
                               VADOracle fires: speech_start=18.5s, end=22.3s
                                         │
                               read_window(18.5, 22.3) → float32 audio
                                         │
                                    Whisper → transcript

Why this fixes first-word clipping
────────────────────────────────────
OLD (VAD-gated): VAD onset fires ~32ms into speech.  Audio before onset is
  lost.  pre-roll ring partially helps but is cleared on state transitions.

NEW: Audio is ALWAYS in the buffer.  VADOracle subtracts pre_buffer_s=2.0
  from the onset timestamp so read_window() starts 2s BEFORE the first
  word, guaranteed to capture it regardless of VAD latency.

Time reference
──────────────
All timestamps are session-elapsed seconds (time.perf_counter() since
buffer creation).  VADOracle uses the same reference via current_time_s.

Thread safety
─────────────
Single writer (_audio_ingestion_loop) + single reader (_run_turn).
Protected by a lock; writes never block.
"""
import threading
import time

import numpy as np


class RollingAudioBuffer:
    """
    Thread-safe circular audio buffer with timestamp-based window reads.

    Capacity: capacity_s × sample_rate samples (default 60 s × 16 kHz = 960 k).
    Memory:   960 000 × 4 bytes ≈ 3.8 MB — trivial.
    Writes:   never block; oldest audio silently overwritten.
    Reads:    read_window(start_s, end_s) returns exactly the requested window,
              clamped to available history.
    """

    def __init__(self, capacity_s: float = 60.0, sample_rate: int = 16_000):
        self._sr   = sample_rate
        self._cap  = int(capacity_s * sample_rate)
        self._buf  = np.zeros(self._cap, dtype=np.float32)
        self._lock = threading.Lock()

        # Ring-write head and total-samples counter.
        # _write_pos advances mod _cap on every write.
        # _total tracks absolute sample count (never wraps) for timestamp math.
        self._write_pos = 0
        self._total     = 0

        # Session start — same perf_counter reference used by VADOracle.
        self._t0 = time.perf_counter()

    # ── Write ──────────────────────────────────────────────────────────────

    def write(self, samples: np.ndarray) -> None:
        """
        Append float32 samples.  Never blocks.
        Oldest audio is silently overwritten when the ring is full.
        """
        if samples is None or len(samples) == 0:
            return
        samples = np.asarray(samples, dtype=np.float32)
        n = len(samples)

        with self._lock:
            end = self._write_pos + n
            if end <= self._cap:
                self._buf[self._write_pos:end] = samples
            else:
                # Wrap: write tail of samples at end of buffer, head at start.
                first = self._cap - self._write_pos
                self._buf[self._write_pos:] = samples[:first]
                self._buf[:n - first]        = samples[first:]

            self._write_pos = (self._write_pos + n) % self._cap
            self._total    += n

    # ── Read ───────────────────────────────────────────────────────────────

    def read_window(self, start_s: float, end_s: float) -> np.ndarray:
        """
        Extract audio for session-elapsed time window [start_s, end_s].

        Both arguments are seconds since buffer creation (same reference as
        current_time_s and the timestamps emitted by VADOracle).

        Returns a float32 ndarray.  The window is clamped to available audio:
          - If start_s is beyond available history (> capacity_s ago), the
            oldest available audio is used instead.
          - If end_s is in the future, the most recent audio is returned.
          - Returns an empty array if the window is fully invalid.

        Example:
            # VAD says speech happened from 18.5 s to 22.3 s:
            audio = buf.read_window(18.5, 22.3)  # → ~60 800 samples @ 16 kHz
        """
        if start_s >= end_s:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            total      = self._total
            write_pos  = self._write_pos

        # Convert session times to absolute sample indices.
        start_sample = int(start_s * self._sr)
        end_sample   = int(end_s   * self._sr)

        # Clamp to available ring history.
        oldest = max(0, total - self._cap)
        start_sample = max(start_sample, oldest)
        end_sample   = min(end_sample,   total)

        n = end_sample - start_sample
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            # Re-read under lock in case a write just happened.
            total     = self._total
            write_pos = self._write_pos

            # Recompute with fresh total.
            oldest       = max(0, total - self._cap)
            start_sample = max(int(start_s * self._sr), oldest)
            end_sample   = min(int(end_s   * self._sr), total)
            n            = end_sample - start_sample
            if n <= 0:
                return np.zeros(0, dtype=np.float32)

            # Map absolute sample index to ring position.
            # buf[(write_pos - (total - k)) % cap] == sample k
            samples_before_end = total - start_sample
            ring_start = (write_pos - samples_before_end) % self._cap

            if ring_start + n <= self._cap:
                return self._buf[ring_start:ring_start + n].copy()
            else:
                first = self._cap - ring_start
                return np.concatenate([
                    self._buf[ring_start:].copy(),
                    self._buf[:n - first].copy(),
                ])

    # ── Time helpers ───────────────────────────────────────────────────────

    @property
    def current_time_s(self) -> float:
        """Session-elapsed seconds — same reference as VADOracle timestamps."""
        return time.perf_counter() - self._t0

    @property
    def total_samples_written(self) -> int:
        """Total samples written since creation (never wraps)."""
        with self._lock:
            return self._total
