import json
from unittest.mock import patch
from datetime import datetime


def test_deep_path_calls_reranker(tmp_db, monkeypatch):
    """Deep path should invoke rerank_memories."""
    import numpy as np
    import sqlite3

    # Populate tmp_db with test memories so DB lookups succeed
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    now = datetime.utcnow().isoformat()
    for i in range(5):
        conn.execute("""
            INSERT INTO memories (id, memory_type, raw_text, summary, importance_score,
                                  confidence_score, affect, session_id, created_at)
            VALUES (?, 'IdeaMemory', ?, ?, 7.0, 0.9, 'neutral', 'test-session', ?)
        """, (f"mem-{i}", f"raw text {i}", f"summary {i}", now))
        conn.execute("""
            INSERT INTO memory_embeddings (memory_id, vector_id, model_name)
            VALUES (?, ?, 'all-MiniLM-L6-v2')
        """, (f"mem-{i}", str(i)))
    conn.commit()
    conn.close()

    reranker_called = [False]

    def mock_reranker(query, candidates):
        reranker_called[0] = True
        return candidates[:3]

    monkeypatch.setattr("memory_system.retrieval.search.generate_embedding_vector",
                        lambda t: np.ones(384, dtype="float32"))
    monkeypatch.setattr("memory_system.retrieval.search.search_vector",
                        lambda v, top_k: ([0.9] * 5, list(range(5))))
    monkeypatch.setattr("memory_system.retrieval.search.rerank_memories", mock_reranker)
    monkeypatch.setattr("memory_system.retrieval.search._call_rewriter_llm",
                        lambda q: q)

    from memory_system.retrieval.search import retrieve_memories
    retrieve_memories("why did we choose FAISS over ChromaDB?", path="deep")
    assert reranker_called[0]


def test_fast_path_skips_reranker(monkeypatch):
    """Fast path should NOT call rerank_memories."""
    import numpy as np

    reranker_called = [False]

    def mock_reranker(query, candidates):
        reranker_called[0] = True
        return candidates[:3]

    monkeypatch.setattr("memory_system.retrieval.search.generate_embedding_vector",
                        lambda t: np.ones(384, dtype="float32"))
    monkeypatch.setattr("memory_system.retrieval.search.search_vector",
                        lambda v, top_k: ([], [-1]))
    monkeypatch.setattr("memory_system.retrieval.search.rerank_memories", mock_reranker)

    from memory_system.retrieval.search import retrieve_memories
    retrieve_memories("hi there", path="fast")
    assert not reranker_called[0]
