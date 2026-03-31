

# stt_adapter.py
"""
STT Adapter for integrating your stt2 WhisperEngine into the main VAD → ASR pipeline.

Purpose:
- Accept in‑memory numpy audio arrays
- Convert them into a format your WhisperEngine can transcribe
- Call WhisperEngine.transcribe(path) internally using a temporary WAV file
- Return a clean, structured result for the pipeline

This avoids rewriting your existing WhisperEngine and keeps pipeline clean.
"""

import numpy as np
import tempfile
import soundfile as sf


class STTAdapter:
    """
    Wraps your stt2 WhisperEngine so it can be used inside the real-time pipeline.

    Expected WhisperEngine API:
        text = whisper_engine.transcribe("file_path.wav")

    Pipeline API we provide:
        result = adapter.transcribe_audio(np_array, sample_rate)
    """

    def __init__(self, whisper_engine):
        self.engine = whisper_engine

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000):
        """
        Converts numpy array audio into a temporary WAV file and sends
        it to your WhisperEngine for transcription.

        Returns:
            {
                "text": str,
                "confidence": None,        # WhisperEngine does not return confidence afaik
                "source": "stt2"
            }
        """
        if audio is None or len(audio) == 0:
            return {
                "text": "",
                "confidence": None,
                "source": "stt2"
            }

        # Ensure float32 format for WAV writing
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        text = ""

        # Save to temp file and call your engine
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            sf.write(tf.name, audio, sample_rate)
            try:
                text = self.engine.transcribe(tf.name)
            except Exception as e:
                print("STT2 WhisperEngine Error:", e)
                text = ""

        return {
            "text": text or "",
            "confidence": None,
            "source": "stt2"
        }


# Local test (does not record mic)
if __name__ == "__main__":
    print("STTAdapter loaded successfully.")