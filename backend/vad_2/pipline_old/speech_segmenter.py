

# speech_segmenter.py
"""
Speech Segmenter for Silero Pipeline

This module takes continuous VAD states from vad_connector and:
    - Starts collecting audio when VAD = speech
    - Stops collecting when VAD = noise
    - Outputs a completed speech segment for ASR + intent classification

It is designed for real-time low-latency interrupt detection.
"""

import numpy as np


class SpeechSegmenter:
    def __init__(self, sample_rate=16000, min_segment_ms=200, max_segment_ms=1500):
        """
        Parameters:
            min_segment_ms → ignore very short bursts (e.g. 50–100ms)
            max_segment_ms → force cutoff (avoid capturing endless speech)
        """
        self.sample_rate = sample_rate
        self.min_samples = int((min_segment_ms / 1000) * sample_rate)
        self.max_samples = int((max_segment_ms / 1000) * sample_rate)

        self.collecting = False
        self.segment_buffer = []

    def reset(self):
        self.collecting = False
        self.segment_buffer = []

    def update(self, vad_state: dict):
        """
        Input vad_state = {
            'prob': float,
            'speech': bool,
            'audio_512': np.ndarray
        }

        Returns:
            (segment or None)
        """
        audio_frame = vad_state["audio_512"]
        speech_flag = vad_state["speech"]

        # Case 1 → start recording
        if speech_flag and not self.collecting:
            self.collecting = True
            self.segment_buffer = []
            self.segment_buffer.append(audio_frame)
            return None

        # Case 2 → continue recording
        if speech_flag and self.collecting:
            self.segment_buffer.append(audio_frame)

            # safety cutoff → segment too long
            total = len(self.segment_buffer) * len(audio_frame)
            if total >= self.max_samples:
                segment = np.concatenate(self.segment_buffer)
                self.reset()
                return segment

            return None

        # Case 3 → speech ended → finalize segment
        if not speech_flag and self.collecting:
            total = len(self.segment_buffer) * len(audio_frame)

            # too short → ignore
            if total < self.min_samples:
                self.reset()
                return None

            # valid segment
            segment = np.concatenate(self.segment_buffer)
            self.reset()
            return segment

        # Case 4 → idle noise
        return None


# Quick self-test
if __name__ == "__main__":
    # Fake audio for testing
    seg = SpeechSegmenter()

    # Simulate speech for 5 frames
    for _ in range(5):
        out = seg.update({"prob": 0.9, "speech": True, "audio_512": np.ones(512)})
        print("Collecting:", out)

    # Simulate silence for 3 frames → should finalize
    for _ in range(3):
        out = seg.update({"prob": 0.1, "speech": False, "audio_512": np.zeros(512)})
        print("Final:", out)