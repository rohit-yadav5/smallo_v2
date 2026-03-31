# vad_connector.py
"""
Connects the SileroBuffer + Silero VAD engine.
This module exposes a clean interface for the pipeline to get:
- live speech probability
- speech/noise state
- raw audio stream (via buffer)

It does NOT make decisions. It only provides:
    VADEngine.get_vad_state() → {probability, is_speech}
"""

import numpy as np
from ..buffer.silero_buffer import SileroBuffer
from ..vad_engine_silero.main import SileroVADEngine


class VADConnector:
    def __init__(self, sample_rate=16000, frame_ms=20):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_size = int(sample_rate * frame_ms / 1000)

        # buffer receives raw mic audio
        self.buffer = SileroBuffer(max_seconds=2.0, sample_rate=sample_rate)

        # Silero VAD engine
        self.vad = SileroVADEngine(sample_rate=sample_rate, threshold=0.5)

    def add_audio(self, audio_frame: np.ndarray):
        """
        Called by microphone stream callback.
        """
        self.buffer.add_frames(audio_frame)

    def get_vad_state(self):
        """
        Returns:
            {
                "prob": float,
                "speech": bool,
                "audio_512": np.ndarray
            }
        """
        audio_512 = self.buffer.get_last_samples(512)
        prob, speech = self.vad.process_frame(audio_512)

        return {
            "prob": prob,
            "speech": speech,
            "audio_512": audio_512
        }
