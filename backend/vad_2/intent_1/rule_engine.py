# classifier/rule_engine.py

from vad.intent_1.constants import INTERRUPT_KEYWORDS, IGNORE_PATTERNS

class RuleEngine:
    def __init__(self):
        self.interrupt_keywords = INTERRUPT_KEYWORDS
        self.ignore_patterns = IGNORE_PATTERNS

    def check(self, text: str):
        text = text.lower().strip()

        # Direct interrupt matches
        for kw in self.interrupt_keywords:
            if kw in text:
                return "INTERRUPT", f"matched keyword: {kw}"

        # Direct ignore matches
        for kw in self.ignore_patterns:
            if kw in text:
                return "IGNORE", f"matched ignore: {kw}"

        # No rule triggered
        return "UNKNOWN", "no keyword matched"
