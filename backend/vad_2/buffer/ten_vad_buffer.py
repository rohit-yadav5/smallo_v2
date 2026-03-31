import numpy as np
import threading

class AudioBuffer:
    def __init__(self, max_seconds=2, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = max_seconds * sample_rate
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.lock = threading.Lock()
        self.write_pos = 0
        self.is_full = False

    def add_frames(self, audio_frames: np.ndarray):
        with self.lock:
            frames_len = len(audio_frames)

            # If frames exceed buffer size, keep only last part
            if frames_len >= self.max_samples:
                self.buffer = audio_frames[-self.max_samples:]
                self.write_pos = 0
                self.is_full = True
                return

            end_pos = self.write_pos + frames_len

            # Wrap-around logic
            if end_pos <= self.max_samples:
                self.buffer[self.write_pos:end_pos] = audio_frames
            else:
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = audio_frames[:first_part]
                self.buffer[:frames_len - first_part] = audio_frames[first_part:]

            self.write_pos = (self.write_pos + frames_len) % self.max_samples

            if self.write_pos == 0:
                self.is_full = True

    def get_last(self, seconds: float):
        samples_needed = int(seconds * self.sample_rate)
        if samples_needed > self.max_samples:
            samples_needed = self.max_samples

        with self.lock:
            if not self.is_full and self.write_pos < samples_needed:
                # Not enough samples yet
                return self.buffer[:self.write_pos].copy()

            start = (self.write_pos - samples_needed) % self.max_samples

            if start + samples_needed <= self.max_samples:
                return self.buffer[start:start + samples_needed].copy()
            else:
                first_part = self.max_samples - start
                return np.concatenate([
                    self.buffer[start:],
                    self.buffer[:samples_needed - first_part]
                ]).copy()
