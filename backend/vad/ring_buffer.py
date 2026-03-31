"""
vad/ring_buffer.py — Thread-safe circular audio ring buffer.

Adapted from vad_2/buffer/silero_buffer.py.

Designed for continuous audio streaming: audio is written from one thread
(WebSocket ingestion), read from another (VAD processing).

Key properties
──────────────
• Fixed capacity (max_seconds × sample_rate samples)
• Writes never block — newest audio always wins (old audio silently overwritten)
• get_last_samples(n) always returns exactly n samples; pads with zeros if
  not enough audio has arrived yet
• Thread-safe via a single lock
"""
import threading
import numpy as np


class RingBuffer:
    """Circular ring buffer tuned for 16 kHz float32 audio."""

    def __init__(self, max_seconds: float = 3.0, sample_rate: int = 16_000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self._buf       = np.zeros(self.max_samples, dtype=np.float32)
        self._lock      = threading.Lock()
        self._write_pos = 0
        self._is_full   = False

    # ── Write ──────────────────────────────────────────────────────────────

    def add_frames(self, audio: np.ndarray) -> None:
        """Append float32 samples into the ring.  Normalises int types."""
        if audio is None or len(audio) == 0:
            return

        if audio.dtype.kind in "iu":
            maxv  = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / maxv
        else:
            audio = audio.astype(np.float32)

        n = len(audio)
        with self._lock:
            if n >= self.max_samples:
                # New audio is larger than buffer — keep only the tail
                self._buf[:] = audio[-self.max_samples:]
                self._write_pos = 0
                self._is_full   = True
                return

            end = self._write_pos + n
            if end <= self.max_samples:
                self._buf[self._write_pos:end] = audio
            else:
                first = self.max_samples - self._write_pos
                self._buf[self._write_pos:] = audio[:first]
                self._buf[:n - first]        = audio[first:]

            self._write_pos = (self._write_pos + n) % self.max_samples
            if self._write_pos == 0:
                self._is_full = True

    # ── Read ───────────────────────────────────────────────────────────────

    def get_last_samples(self, n: int) -> np.ndarray:
        """Return exactly the most-recent n samples (float32, zero-padded if needed)."""
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            n = min(n, self.max_samples)

            if not self._is_full and self._write_pos < n:
                # Not enough real audio yet — left-pad with zeros
                needed = n - self._write_pos
                return np.concatenate([
                    np.zeros(needed, dtype=np.float32),
                    self._buf[:self._write_pos].copy(),
                ])

            start = (self._write_pos - n) % self.max_samples
            if start + n <= self.max_samples:
                return self._buf[start:start + n].copy()
            else:
                first = self.max_samples - start
                return np.concatenate([
                    self._buf[start:].copy(),
                    self._buf[:n - first].copy(),
                ])

    @property
    def filled_samples(self) -> int:
        """How many valid samples are currently in the buffer."""
        with self._lock:
            return self.max_samples if self._is_full else self._write_pos
