# hybrid_classifier.py
"""
A clean HybridIntentClassifier that wraps the rule engine and ML model
located inside vad.intent_1. This file intentionally uses absolute package
imports so that `from vad.intent_1.hybrid_classifier import HybridIntentClassifier`
works when running the project as a package.
"""

from typing import Dict

from vad.intent_1.rule_engine import RuleEngine
from vad.intent_1.ml_model import MLIntentClassifier # type: ignore


class HybridIntentClassifier:
    def __init__(self, ml_model_path: str = "intent-bert", ml_threshold: float = 0.7):
        self.rules = RuleEngine()
        self.ml = MLIntentClassifier(model_path=ml_model_path, threshold=ml_threshold)

    def predict(self, text: str) -> Dict:
        """Return a unified result dictionary.

        Output example:
        {
            'result': 'INTERRUPT'|'IGNORE'|'UNKNOWN',
            'confidence': float,
            'source': 'rule'|'ml',
            'detail': {...}
        }
        """
        if text is None:
            text = ""

        # Rule-first
        rule_out = self.rules.check(text)
        if rule_out.get("intent") == "INTERRUPT":
            return {
                "result": "INTERRUPT",
                "confidence": float(rule_out.get("confidence", 1.0)),
                "source": "rule",
                "detail": rule_out,
            }

        # ML fallback
        ml_out = self.ml.predict(text)
        intent = ml_out.get("intent", "UNKNOWN")
        confidence = float(ml_out.get("confidence", 0.0))

        return {
            "result": intent,
            "confidence": confidence,
            "source": "ml",
            "detail": ml_out,
        }


# quick self-test
if __name__ == "__main__":
    clf = HybridIntentClassifier()
    samples = ["stop", "hello", "please stop now", "umm"]
    for s in samples:
        print(s, "->", clf.predict(s))