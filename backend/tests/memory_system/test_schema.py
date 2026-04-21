def test_affect_column_exists_after_migration(tmp_db):
    import sqlite3
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(memories)")
    columns = {row["name"] for row in cursor.fetchall()}
    conn.close()
    assert "affect" in columns

def test_affect_default_is_neutral(tmp_db):
    import sqlite3
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memories (id, memory_type, raw_text, summary, importance_score)
        VALUES ('test-001', 'IdeaMemory', 'hello world', 'hello', 5.0)
    """)
    conn.commit()
    cursor.execute("SELECT affect FROM memories WHERE id = 'test-001'")
    row = cursor.fetchone()
    conn.close()
    assert row["affect"] == "neutral"
