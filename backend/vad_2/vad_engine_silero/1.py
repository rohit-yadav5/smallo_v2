import sounddevice as sd
import numpy as np

from vad_engine_silero.main import SileroVADEngine
from vad_engine_silero.buffer.silero_buffer import SileroBuffer


class LiveSileroVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        threshold: float = 0.5,
        buffer_seconds: int = 2,
    ):
        # audio config
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_size = int(self.sample_rate * self.frame_ms / 1000)

        # core components
        self.buffer = SileroBuffer(
            max_seconds=buffer_seconds,
            sample_rate=self.sample_rate,
        )
        self.vad = SileroVADEngine(
            sample_rate=self.sample_rate,
            threshold=threshold,
        )

        self.stream = None
        self.running = False

    # ---------- MIC CALLBACK ----------
    def _mic_callback(self, indata, frames, time, status):
        if status:
            print("Mic warning:", status)

        audio = indata[:, 0].copy()
        self.buffer.add_frames(audio)

    # ---------- START ----------
    def start(self):
        self.running = True

        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            callback=self._mic_callback,
        )

        self.stream.start()
        print("Live Silero VAD started (Ctrl+C to stop)")

        try:
            self._run_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ---------- MAIN LOOP ----------
    def _run_loop(self):
        while self.running:
            audio_window = self.buffer.get_last_samples(512)

            if audio_window is None or len(audio_window) == 0:
                sd.sleep(20)
                continue

            prob, is_speech = self.vad.process_frame(audio_window)

            state = "speech ✅" if is_speech else "noise 🔴"
            print(f"{prob:.3f} → {state}")

            sd.sleep(20)

    # ---------- STOP ----------
    def stop(self):
        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()

        print("Live Silero VAD stopped")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    vad = LiveSileroVAD()
    vad.start()