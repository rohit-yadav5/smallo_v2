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


def get_or_create_entity(cursor, name: str, domain: str, category: str, entity_type: str):

    normalized = name.strip().lower()

    cursor.execute("""
        SELECT id, usage_count FROM entities WHERE name = ?
    """, (normalized,))
    row = cursor.fetchone()

    if row:
        entity_id = row["id"]

        # Increment usage_count
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

    # Create new entity
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
        1,              # first usage
        1.0,            # base importance
        datetime.utcnow().isoformat()
    ))

    # NOTE: entity_relations (parent links) are intentionally NOT written here.
    # The audit (memory_audit.md §10) found that entity_relations is never read
    # during retrieval — writing to it was wasted I/O. The table is kept in the
    # schema for future use but no longer populated on every insert.

    return entity_id