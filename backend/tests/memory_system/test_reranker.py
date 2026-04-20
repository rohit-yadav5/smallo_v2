import json

_CANDIDATES = [
    {"memory_id": f"m{i}", "summary": f"summary {i}", "raw_text": f"full text {i}", "score": 0.5}
    for i in range(20)
]


def test_coarse_filter_returns_at_most_10(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.reranker._call_llm_for_ranking",
        lambda prompt: json.dumps(list(range(10)))
    )
    from memory_system.retrieval.reranker import _coarse_filter
    result = _coarse_filter("test query", _CANDIDATES)
    assert len(result) <= 10


def test_fine_rank_returns_at_most_5(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.reranker._call_llm_for_ranking",
        lambda prompt: json.dumps([0, 1, 2, 3, 4])
    )
    from memory_system.retrieval.reranker import _fine_rank
    result = _fine_rank("test query", _CANDIDATES[:10])
    assert len(result) <= 5


def test_rerank_returns_5_on_success(monkeypatch):
    call_n = [0]
    def _mock_llm(prompt):
        call_n[0] += 1
        return json.dumps(list(range(min(10, call_n[0] * 5))))
    monkeypatch.setattr("memory_system.retrieval.reranker._call_llm_for_ranking", _mock_llm)
    monkeypatch.setattr("memory_system.retrieval.reranker.can_load_7b", lambda: False)

    from memory_system.retrieval.reranker import rerank_memories
    result = rerank_memories("what did I do yesterday?", _CANDIDATES)
    assert len(result) <= 5


def test_rerank_falls_back_on_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.reranker._call_llm_for_ranking",
        lambda prompt: "not valid json at all {{{"
    )
    monkeypatch.setattr("memory_system.retrieval.reranker.can_load_7b", lambda: False)
    from memory_system.retrieval.reranker import rerank_memories
    result = rerank_memories("query", _CANDIDATES)
    assert len(result) <= 5
    assert all(c in _CANDIDATES for c in result)


def test_rerank_empty_candidates_returns_empty(monkeypatch):
    monkeypatch.setattr("memory_system.retrieval.reranker.can_load_7b", lambda: False)
    from memory_system.retrieval.reranker import rerank_memories
    assert rerank_memories("query", []) == []


def test_rerank_uses_7b_when_ram_available(monkeypatch):
    """When can_load_7b() returns True, _call_llm_for_ranking should use planner_model."""
    from config.llm import LLM_CONFIG
    used_models = []

    def _mock_ask_llm(prompt, model=None):
        used_models.append(model)
        return json.dumps([0, 1, 2, 3, 4])

    monkeypatch.setattr("memory_system.retrieval.reranker.can_load_7b", lambda: True)
    import llm as llm_module
    monkeypatch.setattr(llm_module, "ask_llm", _mock_ask_llm)

    from memory_system.retrieval.reranker import rerank_memories
    rerank_memories("query", _CANDIDATES[:10])

    assert all(m == LLM_CONFIG.planner_model for m in used_models)


def test_rerank_uses_3b_when_ram_low(monkeypatch):
    """When can_load_7b() returns False, _call_llm_for_ranking should use model (3b)."""
    from config.llm import LLM_CONFIG
    used_models = []

    def _mock_ask_llm(prompt, model=None):
        used_models.append(model)
        return json.dumps([0, 1, 2, 3, 4])

    monkeypatch.setattr("memory_system.retrieval.reranker.can_load_7b", lambda: False)
    import llm as llm_module
    monkeypatch.setattr(llm_module, "ask_llm", _mock_ask_llm)

    from memory_system.retrieval.reranker import rerank_memories
    rerank_memories("query", _CANDIDATES[:10])

    assert all(m == LLM_CONFIG.model for m in used_models)
