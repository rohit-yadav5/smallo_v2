import asyncio


def test_rewrite_returns_string(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.query_rewriter._call_rewriter_llm",
        lambda q: "server configuration deployment backend infrastructure"
    )
    from memory_system.retrieval.query_rewriter import rewrite_query_for_retrieval
    result = asyncio.run(rewrite_query_for_retrieval("what did we say about the server?"))
    assert isinstance(result, str)
    assert len(result) > 5


def test_rewrite_falls_back_on_llm_error(monkeypatch):
    def _raise(q):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(
        "memory_system.retrieval.query_rewriter._call_rewriter_llm", _raise
    )
    from memory_system.retrieval.query_rewriter import rewrite_query_for_retrieval
    original = "what did we decide about auth?"
    result = asyncio.run(rewrite_query_for_retrieval(original))
    assert result == original


def test_rewrite_rejects_empty_llm_response(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.query_rewriter._call_rewriter_llm",
        lambda q: "   "
    )
    from memory_system.retrieval.query_rewriter import rewrite_query_for_retrieval
    original = "what is the database schema?"
    result = asyncio.run(rewrite_query_for_retrieval(original))
    assert result == original
