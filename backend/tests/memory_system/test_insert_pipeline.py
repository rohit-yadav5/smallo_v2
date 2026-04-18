from unittest.mock import patch, MagicMock
import sqlite3


def test_insert_stores_affect_column(tmp_db, monkeypatch):
    """After insert, the affect column should be populated (not NULL or empty)."""
    monkeypatch.setattr("memory_system.core.insert_pipeline.generate_embedding_vector",
                        lambda text: __import__("numpy").zeros(384, dtype="float32"))
    monkeypatch.setattr("memory_system.core.insert_pipeline.search_vector",
                        lambda vec, top_k: ([], []))
    monkeypatch.setattr("memory_system.core.insert_pipeline.add_vector",
                        lambda mid, vec: 0)
    monkeypatch.setattr("memory_system.core.insert_pipeline.extract_entities",
                        lambda text: [])

    from memory_system.core.insert_pipeline import insert_memory
    mid = insert_memory({"text": "I love this feature, it works perfectly!", "memory_type": "IdeaMemory"})

    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT affect FROM memories WHERE id = ?", (mid,)).fetchone()
    conn.close()
    assert row is not None
    assert row["affect"] in {"positive", "negative", "neutral", "frustrated", "excited", "uncertain"}


def test_near_duplicate_creates_chain_link(tmp_db, monkeypatch):
    """Two similar (0.75-0.89) inserts should create a chain relation."""
    import numpy as np
    call_count = [0]

    def mock_search(vec, top_k):
        call_count[0] += 1
        if call_count[0] == 1:
            return ([0.82], [0])   # near-duplicate on first insert
        return ([], [-1])

    monkeypatch.setattr("memory_system.core.insert_pipeline.generate_embedding_vector",
                        lambda text: np.zeros(384, dtype="float32"))
    monkeypatch.setattr("memory_system.core.insert_pipeline.search_vector", mock_search)
    monkeypatch.setattr("memory_system.core.insert_pipeline.add_vector",
                        lambda mid, vec: 0)
    monkeypatch.setattr("memory_system.core.insert_pipeline.extract_entities",
                        lambda text: [])

    # Pre-insert a memory that will act as the near-duplicate
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        INSERT INTO memories (id, memory_type, raw_text, summary, importance_score, affect)
        VALUES ('existing-001', 'IdeaMemory', 'old text', 'old summary', 5.0, 'positive')
    """)
    conn.execute("""
        INSERT INTO memory_embeddings (memory_id, vector_id, model_name)
        VALUES ('existing-001', '0', 'all-MiniLM-L6-v2')
    """)
    conn.commit()
    conn.close()

    from memory_system.core.insert_pipeline import insert_memory
    new_mid = insert_memory({"text": "similar text here", "memory_type": "IdeaMemory"})

    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT * FROM memory_relations WHERE source_memory_id = ?
    """, (new_mid,)).fetchone()
    conn.close()
    assert row is not None
