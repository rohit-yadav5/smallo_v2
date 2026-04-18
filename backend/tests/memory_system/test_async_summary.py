def test_high_signal_phrase_gives_bump():
    from memory_system.core.async_summary import _compute_importance_bump
    assert _compute_importance_bump("I prefer dark mode for all interfaces") == 2.0


def test_neutral_summary_gives_no_bump():
    from memory_system.core.async_summary import _compute_importance_bump
    assert _compute_importance_bump("The server started on port 8765") == 0.0


def test_decision_phrase_gives_bump():
    from memory_system.core.async_summary import _compute_importance_bump
    assert _compute_importance_bump("Decided to use FAISS over ChromaDB") == 2.0


def test_call_llm_for_summary_returns_string(monkeypatch):
    """_call_llm_for_summary returns a non-empty string when LLM responds."""
    import llm as llm_module
    monkeypatch.setattr(llm_module, "ask_llm", lambda prompt: "One concise summary sentence.")
    from memory_system.core.async_summary import _call_llm_for_summary
    result = _call_llm_for_summary("some long text here", "IdeaMemory")
    assert isinstance(result, str)
    assert len(result) > 0


def test_call_llm_for_summary_truncates_long_result(monkeypatch):
    """_call_llm_for_summary truncates LLM output to 300 chars."""
    import llm as llm_module
    monkeypatch.setattr(llm_module, "ask_llm", lambda prompt: "x" * 500)
    from memory_system.core.async_summary import _call_llm_for_summary
    result = _call_llm_for_summary("text", "IdeaMemory")
    assert len(result) <= 300
