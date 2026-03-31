import sounddevice as sd
import numpy as np
from vad.buffer.silero_buffer import SileroBuffer

class AudioListener:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.buffer = SileroBuffer(sample_rate=sample_rate)

    def start(self):
        def callback(indata, frames, time, status):
            if status:
                return
            audio = indata[:, 0].copy()
            self.buffer.add_frames(audio)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
            blocksize=512
        )
        stream.start()
        return stream

    def get_chunk(self, size=512):
        buf = self.buffer.get_buffer()
        if len(buf) < size:
            return None
        return buf[-size:]