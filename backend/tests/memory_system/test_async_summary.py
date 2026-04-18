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
    monkeypatch.setattr("memory_system.core.async_summary._call_llm_for_summary",
                        lambda text, memory_type: "Test summary sentence.")
    from memory_system.core.async_summary import _call_llm_for_summary
    result = _call_llm_for_summary("some long text here", "IdeaMemory")
    assert isinstance(result, str)
    assert len(result) > 0


def test_call_llm_for_summary_truncates_long_result(monkeypatch):
    long_result = "x" * 500
    monkeypatch.setattr("memory_system.core.async_summary._call_llm_for_summary",
                        lambda text, memory_type: long_result[:300])
    from memory_system.core.async_summary import _call_llm_for_summary
    result = _call_llm_for_summary("text", "IdeaMemory")
    assert len(result) <= 300
