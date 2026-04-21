import pytest


def test_detect_affect_positive():
    from memory_system.core.affect import detect_affect
    assert detect_affect("That works perfectly, great result!") == "positive"


def test_detect_affect_negative():
    from memory_system.core.affect import detect_affect
    assert detect_affect("The app crashed with an error again") == "negative"


def test_detect_affect_frustrated():
    from memory_system.core.affect import detect_affect
    assert detect_affect("This keeps failing, it's so frustrating") == "frustrated"


def test_detect_affect_excited():
    from memory_system.core.affect import detect_affect
    assert detect_affect("Finally got it working, this is amazing!") == "excited"


def test_detect_affect_uncertain():
    from memory_system.core.affect import detect_affect
    assert detect_affect("Not sure if this approach is correct, maybe try another") == "uncertain"


def test_detect_affect_unknown_falls_back_to_neutral(monkeypatch):
    """LLM fallback returns neutral when LLM also can't classify."""
    monkeypatch.setattr(
        "memory_system.core.affect._detect_affect_llm",
        lambda text: "neutral"
    )
    from memory_system.core.affect import detect_affect
    assert detect_affect("the quick brown fox") == "neutral"
