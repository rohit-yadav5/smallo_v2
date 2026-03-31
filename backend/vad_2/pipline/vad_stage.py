import numpy as np
from vad.vad_engine_silero.main import SileroVADEngine

class VADStage:
    def __init__(self, threshold=0.5):
        self.vad = SileroVADEngine()
        self.threshold = threshold
        self.in_speech = False
        self.collected = []

    def process(self, audio_chunk):
        prob, is_speech = self.vad.process_frame(audio_chunk)

        if is_speech:
            self.in_speech = True
            self.collected.append(audio_chunk)
            return None

        if self.in_speech:
            # speech ended
            self.in_speech = False
            speech = np.concatenate(self.collected)
            self.collected = []
            return speech

        return None