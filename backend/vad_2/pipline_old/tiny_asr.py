# tiny_asr.py
"""
Tiny ASR module for short speech chunks.
This module is designed to work inside the Silero pipeline:
    - VAD detects speech → Segmenter extracts the chunk → ASR transcribes it.

Uses Whisper tiny.en (fastest + stable for command-style audio).
"""

import numpy as np
from faster_whisper import WhisperModel

import json
import os
from datetime import datetime

STT_JSON_PATH = "/Users/rohit/code/6hats/vad/intent_1/stt.json"

def save_stt_text(text: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text.strip()
    }

    if os.path.exists(STT_JSON_PATH):
        try:
            with open(STT_JSON_PATH, "r") as f:
                data = json.load(f)
        except:
            data = []
    else:
        data = []

    data.append(entry)

    with open(STT_JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)


class TinyASR:
    def __init__(self, model_size="tiny.en", device="cpu"):
        """
        Fastest model recommended for real-time intent detection.
        - tiny.en ~ 15-20ms transcription for 700ms audio.
        """
        self.model = WhisperModel(model_size, device=device)

    def transcribe(self, audio: np.ndarray, sample_rate=16000):
        """
        Input:
            audio: numpy array (float32), 16kHz mono

        Returns:
            {
                "text": str,
                "confidence": float,
                "segments": list
            }
        """
        if audio is None or len(audio) == 0:
            return {
                "text": "",
                "confidence": 0.0,
                "segments": []
            }

        # Whisper expects float32 in [-1,1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Run whisper tiny
        segments, info = self.model.transcribe(
            audio,
            beam_size=1,
            best_of=1,
            language="en"
        )

        text = ""
        conf = 0.0
        seg_list = []

        for seg in segments:
            text += seg.text.strip() + " "
            conf = max(conf, seg.avg_logprob)
            seg_list.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "confidence": seg.avg_logprob
                }
            )

        text = text.strip()

        # Convert logprob to approximate 0–1 scale
        if conf < 0:
            conf = 1 / (1 + np.exp(-conf))

        save_stt_text(text)

        return {
            "text": text,
            "confidence": float(conf),
            "segments": seg_list
        }


# Quick test
if __name__ == "__main__":
    import sounddevice as sd

    print("Speak a short command...")
    sr = 16000
    rec = sd.rec(int(sr * 10.0), samplerate=sr, channels=1)
    sd.wait()

    audio = rec[:, 0]

    asr = TinyASR()
    out = asr.transcribe(audio)

    print("\nASR Output:", out)
    save_stt_text(out["text"])
    print("Saved to stt.json")