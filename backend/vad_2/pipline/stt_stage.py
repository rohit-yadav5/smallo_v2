# vad/pipline/stt_stage.py
from vad.stt2.whisper_engine import WhisperEngine

class STTStage:
    def __init__(self):
        self.engine = WhisperEngine()

    def transcribe(self, audio):
        out = self.engine.transcribe_audio_array(audio)
        return out.get("text", "").strip()