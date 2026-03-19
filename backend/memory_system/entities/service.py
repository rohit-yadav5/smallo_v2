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

    # -----------------------------
    # 🔗 Auto Parent Relation
    # -----------------------------
    parent_name = AUTO_PARENT_RULES.get(normalized)

    if parent_name:

        # Ensure parent entity exists
        cursor.execute("""
            SELECT id FROM entities WHERE name = ?
        """, (parent_name,))
        parent_row = cursor.fetchone()

        if parent_row:
            parent_id = parent_row["id"]
        else:
            # Create parent entity if it doesn't exist
            parent_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO entities
                (id, name, domain, category, entity_type, usage_count, importance_score, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                parent_id,
                parent_name,
                "Engineering",
                "Concept",
                "Parent",
                1,
                1.0,
                datetime.utcnow().isoformat()
            ))

        # Create relation (ignore if already exists)
        cursor.execute("""
            INSERT OR IGNORE INTO entity_relations
            (id, source_entity_id, target_entity_id, relation_type, created_at)
            VALUES (?, ?, ?, 'is_a', ?)
        """, (
            f"rel-{entity_id}-{parent_id}",
            entity_id,
            parent_id,
            datetime.utcnow().isoformat()
        ))

    return entity_id