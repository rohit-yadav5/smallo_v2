

# intent_engine.py
"""
Unified Intent Engine for the pipeline.

Combines:
 - RuleEngine (fast, keyword-based)
 - MLIntentClassifier (learned model)

API:
 - IntentEngine.predict(text: str) -> dict
 - IntentEngine.predict_batch(texts: List[str]) -> List[dict]

Output format (dict):
{
    "intent": "INTERRUPT" | "IGNORE" | "UNKNOWN",
    "confidence": float,          # 0..1
    "reason": str,                # textual reason
    "label_id": Optional[int],
    "label_name": Optional[str],
    "source": "rule" | "ml"
}

"""

from typing import List, Optional
from ..intent_classifier.rule_engine import RuleEngine
from ..intent_classifier.ml_model import MLIntentClassifier


class IntentEngine:
    def __init__(self, ml_model_path: str = "intent-bert", ml_threshold: float = 0.7):
        """Load rule engine and ML classifier.

        Parameters:
            ml_model_path: path to HF model folder (default uses ./intent-bert)
            ml_threshold: confidence threshold for ML-based INTERRUPT decision
        """
        self.rules = RuleEngine()
        # initialize ML model with provided path and threshold
        self.ml = MLIntentClassifier(model_path=ml_model_path, threshold=ml_threshold)

    def predict(self, text: str) -> dict:
        """Predict intent for a single text string.

        Rule engine is applied first. If it returns INTERRUPT, we accept it
        immediately (zero-latency rule).

        Otherwise we forward to ML classifier and return its verdict.
        """
        # Defensive handling
        if text is None:
            text = ""

        # 1) fast rule check
        rule_out = self.rules.check(text)
        if rule_out.get("intent") == "INTERRUPT":
            return {
                "intent": rule_out["intent"],
                "confidence": float(rule_out.get("confidence", 1.0)),
                "reason": rule_out.get("reason", "matched interrupt keyword"),
                "label_id": None,
                "label_name": rule_out.get("keyword"),
                "source": "rule"
            }

        # 2) ML fallback
        ml_out = self.ml.predict(text)

        # unify structure
        return {
            "intent": ml_out.get("intent", "UNKNOWN"),
            "confidence": float(ml_out.get("confidence", 0.0)),
            "reason": ml_out.get("reason", "ml_predicted"),
            "label_id": ml_out.get("label_id"),
            "label_name": ml_out.get("label_name"),
            "source": "ml"
        }

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Run predictions for a batch of texts.

        Rule-checks are still done per-item; ML is called per-item using the ML class.
        (If you need heavy batching for ML, modify MLIntentClassifier to accept batches.)
        """
        results = []
        for t in texts:
            results.append(self.predict(t))
        return results


# quick manual test
if __name__ == "__main__":
    engine = IntentEngine()

    samples = [
        "stop",
        "wait a second, i want to say something",
        "hello how are you",
        "umm",
        "stop it now please",
    ]

    for s in samples:
        out = engine.predict(s)
        print(s, "=>", out)