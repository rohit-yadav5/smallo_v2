# pipeline_runner.py
"""
Full real-time pipeline runner:

MIC → VADConnector → SpeechSegmenter → TinyASR → IntentEngine → DecisionEngine

This ties the entire system together into one working loop.
"""

import sounddevice as sd

from .vad_connector import VADConnector
from .speech_segmenter import SpeechSegmenter
from ..intent_classifier.rule_engine import RuleEngine
from ..intent_classifier.ml_model import MLIntentClassifier
from ..stt2.whisper_engine import WhisperEngine
from ..stt2.stt_adapter import STTAdapter


class IntentEngine:
    """
    Combines RuleEngine + MLIntentClassifier
    Returns structured intent output.
    """
    def __init__(self):
        self.rules = RuleEngine()
        self.ml = MLIntentClassifier()

    def predict(self, text: str):
        # Step 1 — rule engine first (fast)
        rule_out = self.rules.check(text)
        if rule_out["intent"] == "INTERRUPT":
            return {
                **rule_out,
                "source": "rule"
            }

        # Step 2 — ML classifier
        ml_out = self.ml.predict(text)
        return {
            **ml_out,
            "source": "ml"
        }


class DecisionEngine:
    """
    Final decision logic for interrupt detection.
    """
    def __init__(self):
        pass

    def decide(self, intent_result: dict):
        """
        intent_result format:
        {
            "intent": "INTERRUPT"/"IGNORE"/"UNKNOWN",
            "confidence": float,
            "source": "rule"/"ml"
        }
        """
        if intent_result["intent"] == "INTERRUPT":
            return True  # interrupt bot
        return False


class PipelineRunner:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

        # Core components
        self.vad = VADConnector(sample_rate=sample_rate)
        self.segmenter = SpeechSegmenter(sample_rate=sample_rate)
        self.whisper_engine = WhisperEngine()
        self.stt = STTAdapter(self.whisper_engine)
        self.intent_engine = IntentEngine()
        self.decision = DecisionEngine()

        # Mic
        self.frame_size = int(sample_rate * 0.02)  # 20ms frames

    def _mic_callback(self, indata, frames, time, status):
        if status:
            print("Mic warning:", status)

        audio_frame = indata[:, 0].copy()
        self.vad.add_audio(audio_frame)

    def run(self):
        stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            callback=self._mic_callback
        )

        stream.start()
        print("\nPipeline Started… (Ctrl+C to stop)\n")

        try:
            while True:
                # Get VAD state
                vad_state = self.vad.get_vad_state()

                # Send to segmenter
                segment = self.segmenter.update(vad_state)

                # If speech segment completed → run ASR + Intent
                if segment is not None:
                    print("\n[Segment Completed] Running ASR…")
                    asr_out = self.stt.transcribe_audio(segment, sample_rate=self.sample_rate)

                    text = asr_out["text"]
                    print("ASR Text:", text)

                    intent_out = self.intent_engine.predict(text)
                    print("Intent:", intent_out)

                    # Final decision
                    if self.decision.decide(intent_out):
                        print("\n🔥 INTERRUPT TRIGGERED! 🔥\n")
                    else:
                        print("\n(no interrupt)\n")

                sd.sleep(10)

        except KeyboardInterrupt:
            print("Stopping pipeline…")

        finally:
            stream.stop()
            stream.close()


if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()