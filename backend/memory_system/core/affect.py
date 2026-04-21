"""Affect tagging module - classifies emotional tone of text."""

_FRUSTRATED = {"frustrated", "annoying", "keeps", "keep failing", "why won't", "ugh"}
_EXCITED = {"amazing", "awesome", "finally", "breakthrough", "excited", "wow"}
_POSITIVE = {"great", "love", "perfect", "excellent", "happy", "solved", "fixed", "success", "works", "working"}
_NEGATIVE = {"hate", "broken", "failed", "error", "crash", "terrible", "wrong", "bad", "issue", "bug"}
_UNCERTAIN = {"maybe", "not sure", "unsure", "unclear", "confused", "might", "possibly", "perhaps"}

# Priority order: check more specific emotions first, then broader ones
_PRIORITY = [
    ("frustrated", _FRUSTRATED),
    ("excited", _EXCITED),
    ("negative", _NEGATIVE),
    ("positive", _POSITIVE),
    ("uncertain", _UNCERTAIN),
]


def detect_affect(text: str) -> str:
    """Return emotional affect label for text. Keyword-first, LLM fallback.

    Args:
        text: Input text to classify

    Returns:
        One of: positive, negative, neutral, frustrated, excited, uncertain
    """
    lower = text.lower()
    for label, keywords in _PRIORITY:
        if any(kw in lower for kw in keywords):
            return label
    return _detect_affect_llm(text)


def _detect_affect_llm(text: str) -> str:
    """LLM fallback when no keyword matches. Returns one of the 6 affect labels."""
    try:
        from llm import ask_llm
        prompt = (
            "Classify the emotional affect of this text. "
            "Respond with exactly one word from: positive, negative, neutral, frustrated, excited, uncertain\n\n"
            f"Text: {text[:200]}\n\nAffect:"
        )
        result = ask_llm(prompt).strip().lower().split()[0]
        valid = {"positive", "negative", "neutral", "frustrated", "excited", "uncertain"}
        return result if result in valid else "neutral"
    except Exception:
        return "neutral"
