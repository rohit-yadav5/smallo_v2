# vad/pipline/main.py
import time
from vad.pipline.audio_listener import AudioListener
from vad.pipline.vad_stage import VADStage
from vad.pipline.stt_stage import STTStage
from vad.pipline.intent_stage import IntentStage
from vad.pipline.decision_engine import DecisionEngine

def main():
    listener = AudioListener()
    vad = VADStage()
    stt = STTStage()
    intent = IntentStage()
    decision = DecisionEngine()

    stream = listener.start()
    print("Pipeline running… (Ctrl+C to stop)")

    try:
        while True:
            chunk = listener.get_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue

            speech = vad.process(chunk)
            if speech is None:
                continue

            text = stt.transcribe(speech)
            if not text:
                continue

            intent_result = intent.classify(text)
            decision.decide(intent_result, text)

    except KeyboardInterrupt:
        print("\nStopping pipeline…")
        stream.stop()

if __name__ == "__main__":
    main()