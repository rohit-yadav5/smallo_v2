import sqlite3
import pytest


def _seed_memory(cursor, memory_id: str, affect: str = "neutral"):
    cursor.execute("""
        INSERT OR IGNORE INTO memories (id, memory_type, raw_text, summary, importance_score, affect)
        VALUES (?, 'IdeaMemory', 'test text', 'test', 5.0, ?)
    """, (memory_id, affect))


def test_create_chain_inserts_relation(tmp_db):
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _seed_memory(cursor, "mem-a")
    _seed_memory(cursor, "mem-b")
    conn.commit()

    from memory_system.core.chain import create_chain
    create_chain(cursor, "mem-a", "mem-b", "caused_by")
    conn.commit()

    cursor.execute("""
        SELECT * FROM memory_relations WHERE source_memory_id = 'mem-a' AND relation_type = 'caused_by'
    """)
    assert cursor.fetchone() is not None
    conn.close()


def test_get_chain_links_returns_targets(tmp_db):
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _seed_memory(cursor, "mem-x")
    _seed_memory(cursor, "mem-y")
    _seed_memory(cursor, "mem-z")
    conn.commit()

    from memory_system.core.chain import create_chain, get_chain_links
    create_chain(cursor, "mem-x", "mem-y", "led_to")
    create_chain(cursor, "mem-x", "mem-z", "confirms")
    conn.commit()

    links = get_chain_links(cursor, "mem-x")
    target_ids = set(links)
    conn.close()
    assert target_ids == {"mem-y", "mem-z"}


def test_get_chain_links_filtered_by_type(tmp_db):
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _seed_memory(cursor, "mem-1")
    _seed_memory(cursor, "mem-2")
    _seed_memory(cursor, "mem-3")
    conn.commit()

    from memory_system.core.chain import create_chain, get_chain_links
    create_chain(cursor, "mem-1", "mem-2", "confirms")
    create_chain(cursor, "mem-1", "mem-3", "contradicts")
    conn.commit()

    # outgoing from mem-1; filter manually since direction param replaced relation_type
    all_links = get_chain_links(cursor, "mem-1", direction="outgoing")
    conn.close()
    assert "mem-2" in all_links
    assert "mem-3" in all_links


@pytest.mark.parametrize("src_affect,tgt_affect,expected", [
    ("positive", "negative", "contradicts"),
    ("excited",  "frustrated", "contradicts"),
    ("positive", "positive", "confirms"),
    ("neutral",  "negative", "confirms"),
    ("frustrated", "frustrated", "confirms"),
])
def test_detect_chain_type(src_affect, tgt_affect, expected):
    from memory_system.core.chain import detect_chain_type
    assert detect_chain_type(src_affect, tgt_affect) == expected


def test_invalid_relation_type_raises():
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(":memory:")
    cursor = conn.cursor()
    from memory_system.core.chain import create_chain
    with pytest.raises(ValueError):
        create_chain(cursor, "a", "b", "invented_type")


def test_get_chain_links_incoming(tmp_db):
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _seed_memory(cursor, "mem-a")
    _seed_memory(cursor, "mem-b")
    conn.commit()

    from memory_system.core.chain import create_chain, get_chain_links
    create_chain(cursor, "mem-b", "mem-a", "caused_by")
    conn.commit()

    incoming = get_chain_links(cursor, "mem-a", direction="incoming")
    conn.close()
    assert incoming == ["mem-b"]


def test_get_chain_links_both_directions(tmp_db):
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _seed_memory(cursor, "mem-a")
    _seed_memory(cursor, "mem-b")
    _seed_memory(cursor, "mem-c")
    conn.commit()

    from memory_system.core.chain import create_chain, get_chain_links
    create_chain(cursor, "mem-a", "mem-b", "led_to")
    create_chain(cursor, "mem-c", "mem-a", "caused_by")
    conn.commit()

    both = get_chain_links(cursor, "mem-a", direction="both")
    conn.close()
    assert set(both) == {"mem-b", "mem-c"}


def test_no_duplicate_chain_on_double_create(tmp_db):
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _seed_memory(cursor, "mem-x")
    _seed_memory(cursor, "mem-y")
    conn.commit()

    from memory_system.core.chain import create_chain
    create_chain(cursor, "mem-x", "mem-y", "confirms")
    conn.commit()
    create_chain(cursor, "mem-x", "mem-y", "confirms")
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM memory_relations WHERE source_memory_id = 'mem-x' AND target_memory_id = 'mem-y'")
    count = cursor.fetchone()[0]
    conn.close()
    assert count == 1
