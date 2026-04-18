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


def test_generate_and_store_summary_updates_db(tmp_db, monkeypatch):
    """generate_and_store_summary should update summary and importance in DB."""
    import asyncio
    import sqlite3
    import llm as llm_module
    monkeypatch.setattr(llm_module, "ask_llm", lambda prompt: "I prefer this approach.")

    # Pre-insert a memory
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        INSERT INTO memories (id, memory_type, raw_text, summary, importance_score, affect)
        VALUES ('sum-001', 'IdeaMemory', 'raw text here', 'placeholder', 5.0, 'neutral')
    """)
    conn.commit()
    conn.close()

    from memory_system.core.async_summary import generate_and_store_summary
    asyncio.run(generate_and_store_summary("sum-001", "raw text here", "IdeaMemory"))

    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT summary, importance_score FROM memories WHERE id = 'sum-001'").fetchone()
    conn.close()

    assert row["summary"] == "I prefer this approach."
    assert row["importance_score"] == 7.0  # 5.0 + 2.0 bump for "i prefer"
