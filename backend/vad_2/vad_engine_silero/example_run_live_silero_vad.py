import sounddevice as sd
import numpy as np

from .main import SileroVADEngine
from ..buffer.silero_buffer import SileroBuffer


class LiveSileroVADRunner:
    def __init__(self, sample_rate=16000, frame_ms=20, threshold=0.8):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_size = int(self.sample_rate * self.frame_ms / 1000)

        self.buffer = SileroBuffer(max_seconds=2, sample_rate=self.sample_rate)
        self.vad = SileroVADEngine(sample_rate=self.sample_rate, threshold=threshold)

        self.running = False
        self.last_state = None

    def _mic_callback(self, indata, frames, time, status):
        if status:
            print("Mic warning:", status)
        audio = indata[:, 0].copy()
        self.buffer.add_frames(audio)

    def start(self):
        self.running = True
        print("Default input device:", sd.default.device)
        print("Default samplerate:", sd.query_devices(sd.default.device[0])["default_samplerate"])

        stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            callback=self._mic_callback,
        )

        stream.start()
        print("Live Silero VAD running… (Ctrl+C to stop)")

        try:
            while self.running:
                audio_window = self.buffer.get_last_samples(512)

                if audio_window is not None and len(audio_window) > 0:
                    prob, speech = self.vad.process_frame(audio_window)
                    current_state = "speech ✅" if speech else "noise 🔴"
                    print(f"{round(prob,3)}  →  {current_state}")

                sd.sleep(20)

        except KeyboardInterrupt:
            print("Stopping…")

        finally:
            stream.stop()
            stream.close()


if __name__ == "__main__":
    runner = LiveSileroVADRunner()
    runner.start()
