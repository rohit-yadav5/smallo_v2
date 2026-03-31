from vad.intent_1.rule_engine import RuleEngine
from vad.intent_1.ml_model import MLIntentClassifier

class HybridIntentClassifier:
    def __init__(self, ml_threshold=0.70):
        self.rules = RuleEngine()
        self.ml = MLIntentClassifier()
        self.ml_threshold = ml_threshold

    def predict(self, text: str):
        text = text.lower().strip()

        # Pre-filter (high precision block)
        safety_filters = [
            "ignore this",
            "ignore that",
            "ignore it",
            "i am ",
            "i'm ",
            "he is ",
            "she is ",
            "they are ",
            "you know",
            "bro",
            "how are you",
            "can you hear me",
            "i am pregnant",
            "let's ignore",
            "ignore this part",
            "ignore that part"
        ]

        for kw in safety_filters:
            if kw in text:
                return {
                    "result": "IGNORE",
                    "source": "pre-filter",
                    "reason": f"matched safety rule: {kw}",
                    "confidence": 1.0,
                    "text": text
                }

        # Rule check
        rule_label, rule_reason = self.rules.check(text)
        if rule_label in ["INTERRUPT", "IGNORE"]:
            return {
                "result": rule_label,
                "source": "rule",
                "reason": rule_reason,
                "confidence": 1.0,
                "text": text
            }

        # ML model
        ml_out = self.ml.predict(text)
        ml_label = ml_out["label"]
        ml_conf = ml_out["confidence"]

        if ml_conf >= max(self.ml_threshold, 0.75):
            return {
                "result": ml_label,
                "source": "ml",
                "confidence": ml_conf,
                "reason": "high confidence",
                "text": text
            }

        return {
            "result": "IGNORE",
            "source": "ml",
            "confidence": ml_conf,
            "reason": "low confidence",
            "text": text
        }