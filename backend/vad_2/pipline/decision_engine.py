# vad/pipline/decision_engine.py
class DecisionEngine:
    def decide(self, intent_result, text):
        intent = intent_result.get("result", "UNKNOWN")

        if intent == "INTERRUPT":
            print(f"[INTERRUPT]: {text}")
            return "INTERRUPT"

        print(f"[IGNORE]: {text}")
        return "IGNORE"