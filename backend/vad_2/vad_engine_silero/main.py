import numpy as np
import torch
from silero_vad import load_silero_vad


class SileroVADEngine:
    """
    Clean and stable OOP Silero VAD engine.

    Methods:
        process_frame(np.ndarray) -> (probability, is_speech)
        run_on_buffer(audio_buffer, callback, window_seconds, period)
    """

    def __init__(self, sample_rate=16000, threshold=0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold

        # Load official Silero VAD model
        self.model = load_silero_vad()
        self.model.eval()

        # Silero minimum recommended chunk size
        self.min_chunk_samples = 512  # Silero requires EXACT 512 samples at 16kHz

    def _prepare_audio(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert numpy PCM -> float32 torch tensor for Silero.
        """

        frame = np.asarray(frame)

        # Convert int16 -> float32 [-1,1]
        if frame.dtype.kind in "iu":
            maxv = float(np.iinfo(frame.dtype).max)
            frame = frame.astype(np.float32) / maxv
        else:
            frame = frame.astype(np.float32)

        return torch.from_numpy(frame).to(torch.float32)

    def process_frame(self, frame: np.ndarray):
        """
        Process a frame and return:
            (probability, is_speech)
        """

        # Silero expects EXACT 512 samples; SileroBuffer guarantees this

        audio_tensor = self._prepare_audio(frame)
        prob = float(self.model(audio_tensor, self.sample_rate).item())
        return prob, prob >= self.threshold

    def run_on_buffer(self, audio_buffer, callback, window_seconds=0.2, period=0.02):
        """
        Poll AudioBuffer and continuously process audio in streaming mode.

        callback(prob, is_speech, frame_audio)
        """

        import time

        while True:
            # Always request exactly 512 samples for Silero
            audio = audio_buffer.get_last_samples(512)

            if audio is not None and len(audio) > 0:
                prob, speech = self.process_frame(audio)
                callback(prob, speech, audio)

            time.sleep(period)


if __name__ == "__main__":
    import sounddevice as sd

    sr = 16000
    engine = SileroVADEngine(sample_rate=sr)

    print("Recording 1 second... Speak now.")
    rec = sd.rec(sr, samplerate=sr, channels=1)
    sd.wait()

    audio = rec[:, 0].astype(np.float32)

    # Silero requires EXACT 512 samples, so test on the first 512-sample chunk
    if len(audio) < engine.min_chunk_samples:
        raise ValueError("Recorded audio too short for Silero VAD demo")

    chunk = audio[:engine.min_chunk_samples]
    prob, speech = engine.process_frame(chunk)

    print("VAD:", round(prob, 3), "Speech" if speech else "Noise")
