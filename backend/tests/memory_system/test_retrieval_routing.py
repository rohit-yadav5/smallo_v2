def test_short_query_routes_fast():
    from memory_system.retrieval.search import classify_retrieval_path
    assert classify_retrieval_path("hello there") == "fast"


def test_long_query_routes_deep():
    from memory_system.retrieval.search import classify_retrieval_path
    assert classify_retrieval_path("can you tell me what we discussed about authentication") == "deep"


def test_question_word_routes_deep():
    from memory_system.retrieval.search import classify_retrieval_path
    assert classify_retrieval_path("why did we choose FAISS?") == "deep"


def test_remember_routes_deep():
    from memory_system.retrieval.search import classify_retrieval_path
    assert classify_retrieval_path("remember what I said about the server") == "deep"


def test_effective_importance_used_in_scoring(monkeypatch):
    import numpy as np
    from datetime import datetime, timedelta

    fresh_created = datetime.utcnow().isoformat()
    old_created   = (datetime.utcnow() - timedelta(days=60)).isoformat()

    from memory_system.core.importance import calculate_effective_importance
    fresh_eff = calculate_effective_importance(7.0, "ActionMemory", fresh_created)
    old_eff   = calculate_effective_importance(7.0, "ActionMemory", old_created)
    assert fresh_eff > old_eff


def test_low_confidence_memories_excluded(tmp_db, monkeypatch):
    import sqlite3, numpy as np
    from datetime import datetime

    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        INSERT INTO memories (id, memory_type, raw_text, summary, importance_score,
                              confidence_score, affect, session_id, created_at)
        VALUES ('low-conf', 'IdeaMemory', 'low confidence memory', 'low conf', 8.0,
                0.2, 'neutral', 'test-session', ?)
    """, (datetime.utcnow().isoformat(),))
    conn.execute("""
        INSERT INTO memory_embeddings (memory_id, vector_id, model_name)
        VALUES ('low-conf', '0', 'all-MiniLM-L6-v2')
    """)
    conn.commit()
    conn.close()

    monkeypatch.setattr("memory_system.retrieval.search.generate_embedding_vector",
                        lambda t: np.ones(384, dtype="float32"))
    monkeypatch.setattr("memory_system.retrieval.search.search_vector",
                        lambda v, top_k: ([0.99], [0]))

    from memory_system.retrieval.search import retrieve_memories
    results = retrieve_memories("low confidence memory", top_k=5)
    result_ids = [r["memory_id"] for r in results]
    assert "low-conf" not in result_ids
