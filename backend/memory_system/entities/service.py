import uuid
from datetime import datetime

AUTO_PARENT_RULES = {
    "redis": "cache",
    "postgresql": "database",
    "postgres": "database",
    "mysql": "database",
    "faiss": "vectordb",
    "chromadb": "vectordb",
    "docker": "infrastructure",
    "kafka": "message_broker"
}


def _write_entity_relation(cursor, child_id: str, parent_name: str) -> None:
    """Write a parent is_a relation if the parent entity already exists."""
    cursor.execute("SELECT id FROM entities WHERE name = ?", (parent_name,))
    row = cursor.fetchone()
    if not row:
        return
    parent_id = row["id"]
    rel_id = f"rel-{child_id[:8]}-{parent_id[:8]}"
    cursor.execute("""
        INSERT OR IGNORE INTO entity_relations
        (id, source_entity_id, target_entity_id, relation_type, created_at)
        VALUES (?, ?, ?, 'is_a', ?)
    """, (rel_id, child_id, parent_id, datetime.utcnow().isoformat()))


def get_or_create_entity(cursor, name: str, domain: str, category: str, entity_type: str):

    normalized = name.strip().lower()

    cursor.execute("""
        SELECT id, usage_count FROM entities WHERE name = ?
    """, (normalized,))
    row = cursor.fetchone()

    if row:
        entity_id = row["id"]
        cursor.execute("""
            UPDATE entities
            SET usage_count = usage_count + 1,
                last_used_at = ?
            WHERE id = ?
        """, (
            datetime.utcnow().isoformat(),
            entity_id
        ))
        return entity_id

    entity_id = str(uuid.uuid4())

    cursor.execute("""
        INSERT INTO entities
        (id, name, domain, category, entity_type, usage_count, importance_score, last_used_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entity_id,
        normalized,
        domain,
        category,
        entity_type,
        1,
        1.0,
        datetime.utcnow().isoformat()
    ))

    parent_name = AUTO_PARENT_RULES.get(normalized)
    if parent_name:
        _write_entity_relation(cursor, entity_id, parent_name)

    return entity_id