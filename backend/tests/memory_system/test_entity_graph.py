import sqlite3


def test_entity_relation_created_for_known_parent(tmp_db):
    """Inserting 'redis' entity should auto-create an is_a relation to 'cache'."""
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Pre-create the parent entity
    cursor.execute("""
        INSERT OR IGNORE INTO entities (id, name, domain, category, entity_type, usage_count, importance_score)
        VALUES ('parent-001', 'cache', 'Engineering', 'Technology', 'Technology', 1, 1.0)
    """)
    conn.commit()

    from memory_system.entities.service import get_or_create_entity
    redis_id = get_or_create_entity(cursor, "redis", "Engineering", "Technology", "Technology")
    conn.commit()

    cursor.execute("""
        SELECT * FROM entity_relations
        WHERE source_entity_id = ? AND target_entity_id = 'parent-001'
    """, (redis_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["relation_type"] == "is_a"


def test_no_relation_created_when_parent_missing(tmp_db):
    """If parent entity doesn't exist yet, no relation is written (no orphan creation)."""
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    from memory_system.entities.service import get_or_create_entity
    # "kafka" → "message_broker" but message_broker entity doesn't exist
    kafka_id = get_or_create_entity(cursor, "kafka", "Engineering", "Infrastructure", "Infrastructure")
    conn.commit()

    cursor.execute("SELECT * FROM entity_relations WHERE source_entity_id = ?", (kafka_id,))
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 0


def test_no_duplicate_relations_on_second_insert(tmp_db):
    """Re-inserting same entity should not duplicate entity_relations rows."""
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO entities (id, name, domain, category, entity_type, usage_count, importance_score)
        VALUES ('parent-db', 'database', 'Engineering', 'Technology', 'Technology', 1, 1.0)
    """)
    conn.commit()

    from memory_system.entities.service import get_or_create_entity
    get_or_create_entity(cursor, "postgresql", "Engineering", "Technology", "Technology")
    conn.commit()
    get_or_create_entity(cursor, "postgresql", "Engineering", "Technology", "Technology")
    conn.commit()

    cursor.execute("SELECT COUNT(*) as cnt FROM entity_relations WHERE relation_type = 'is_a'")
    count = cursor.fetchone()["cnt"]
    conn.close()

    assert count == 1
