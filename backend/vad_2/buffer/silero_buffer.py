import numpy as np
import threading


class SileroBuffer:
    """A simple circular buffer tailored for Silero VAD.

    - Fixed sample rate (default 16 kHz)
    - Designed to return exactly `n` samples on demand via get_last_samples(n)
    - Thread-safe (uses a lock for writes/reads)
    """

    def __init__(self, max_seconds: float = 2.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.lock = threading.Lock()
        self.write_pos = 0
        self.is_full = False

    def add_frames(self, audio_frames: np.ndarray):
        """Append audio frames into the circular buffer.

        audio_frames: 1-D numpy array (float32 or int16)
        """
        if audio_frames is None or len(audio_frames) == 0:
            return

        # normalize int types to float32 in [-1,1]
        if audio_frames.dtype.kind in "iu":
            maxv = float(np.iinfo(audio_frames.dtype).max)
            frames = audio_frames.astype(np.float32) / maxv
        else:
            frames = audio_frames.astype(np.float32)

        frames_len = len(frames)

        with self.lock:
            if frames_len >= self.max_samples:
                # keep only last part
                self.buffer[:] = frames[-self.max_samples:]
                self.write_pos = 0
                self.is_full = True
                return

            end_pos = self.write_pos + frames_len
            if end_pos <= self.max_samples:
                self.buffer[self.write_pos:end_pos] = frames
            else:
                first = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = frames[:first]
                self.buffer[: frames_len - first] = frames[first:]

            self.write_pos = (self.write_pos + frames_len) % self.max_samples
            if self.write_pos == 0:
                self.is_full = True

    def get_last_samples(self, n_samples: int) -> np.ndarray:
        """Return exactly n_samples from the circular buffer (float32).

        - If buffer has less than n_samples of valid audio, pad with zeros on the left.
        - If n_samples > max_samples, n_samples will be clamped to max_samples.
        """
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        with self.lock:
            if n_samples > self.max_samples:
                n_samples = self.max_samples

            # not enough writes yet
            if not self.is_full and self.write_pos < n_samples:
                needed = n_samples - self.write_pos
                out = np.concatenate([
                    np.zeros(needed, dtype=np.float32),
                    self.buffer[: self.write_pos].copy(),
                ])
                return out.astype(np.float32)

            start = (self.write_pos - n_samples) % self.max_samples
            if start + n_samples <= self.max_samples:
                return self.buffer[start : start + n_samples].copy().astype(np.float32)
            else:
                first = self.max_samples - start
                out = np.concatenate([
                    self.buffer[start:].copy(),
                    self.buffer[: n_samples - first].copy(),
                ])
                return out.astype(np.float32)


# Quick self-test when running this file directly
if __name__ == "__main__":
    import sounddevice as sd

    sr = 16000
    buf = SileroBuffer(max_seconds=1.0, sample_rate=sr)

    print("Recording 0.5s and adding to buffer... speak now")
    rec = sd.rec(int(0.5 * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = rec[:, 0]

    buf.add_frames(audio)

    last512 = buf.get_last_samples(512)
    print("Requested 512 samples, got:", len(last512))

    last3200 = buf.get_last_samples(3200)
    print("Requested 3200 samples (clamped), got:", len(last3200))
