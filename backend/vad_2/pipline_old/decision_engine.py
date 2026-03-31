

# decision_engine.py
"""
Final decision-maker for the entire VAD → ASR → Intent pipeline.

Input (from IntentEngine):
{
    "intent": "INTERRUPT" | "IGNORE" | "UNKNOWN",
    "confidence": float,
    "reason": str,
    "source": "rule" | "ml",
    ...
}

Output:
    True  → interrupt the bot
    False → continue normal bot behavior
"""

class DecisionEngine:
    def __init__(self, min_confidence=0.6):
        """
        min_confidence:
            Minimum ML confidence needed to allow ML-based interrupts.
            Rule-based interrupts always bypass this threshold.
        """
        self.min_confidence = min_confidence

    def decide(self, intent_result: dict) -> bool:
        """
        Decide whether the bot should be interrupted based on intent.

        Rule logic:
        - If rule engine says INTERRUPT → always interrupt (immediate).
        - If ML says INTERRUPT → interrupt only if confidence >= threshold.
        - Otherwise → no interrupt.
        """

        if not intent_result:
            return False

        intent = intent_result.get("intent", "UNKNOWN")
        conf = float(intent_result.get("confidence", 0.0))
        source = intent_result.get("source", "ml")

        # RULE ENGINE INTERRUPTS ALWAYS WIN
        if source == "rule" and intent == "INTERRUPT":
            return True

        # ML interrupt needs threshold
        if source == "ml" and intent == "INTERRUPT":
            return conf >= self.min_confidence

        # No interrupt
        return False


# local smoke test
if __name__ == "__main__":
    dec = DecisionEngine(min_confidence=0.6)

    tests = [
        {"intent": "INTERRUPT", "confidence": 1.0, "source": "rule"},
        {"intent": "INTERRUPT", "confidence": 0.8, "source": "ml"},
        {"intent": "INTERRUPT", "confidence": 0.4, "source": "ml"},
        {"intent": "IGNORE", "confidence": 0.9, "source": "ml"},
        {"intent": "UNKNOWN", "confidence": 0.1, "source": "ml"},
    ]

    for t in tests:
        print(t, "→", dec.decide(t))