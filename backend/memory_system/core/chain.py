import uuid
from datetime import datetime

CHAIN_RELATION_TYPES = {"caused_by", "led_to", "contradicts", "confirms", "summarized_by"}

_POSITIVE_AFFECTS = {"positive", "excited"}
_NEGATIVE_AFFECTS = {"negative", "frustrated"}


def create_chain(cursor, source_id: str, target_id: str, relation_type: str) -> str:
    """Insert a directed memory chain link. Returns the relation id."""
    if relation_type not in CHAIN_RELATION_TYPES:
        raise ValueError(
            f"Unknown relation type '{relation_type}'. Must be one of {CHAIN_RELATION_TYPES}"
        )
    cursor.execute("""
        SELECT id FROM memory_relations
        WHERE source_memory_id = ? AND target_memory_id = ? AND relation_type = ?
    """, (source_id, target_id, relation_type))
    existing = cursor.fetchone()
    if existing:
        return existing["id"] if hasattr(existing, "__getitem__") else existing[0]
    rel_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO memory_relations
        (id, source_memory_id, target_memory_id, relation_type, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (rel_id, source_id, target_id, relation_type, datetime.utcnow().isoformat()))
    return rel_id


def get_chain_links(cursor, memory_id: str, direction: str = "outgoing") -> list[dict]:
    """Return linked memory ids based on direction.

    direction="outgoing" — target_memory_id values where source_memory_id = memory_id
    direction="incoming" — source_memory_id values where target_memory_id = memory_id
    direction="both"     — union of outgoing + incoming (deduped)
    """
    if direction == "outgoing":
        cursor.execute("""
            SELECT target_memory_id AS memory_id FROM memory_relations
            WHERE source_memory_id = ?
        """, (memory_id,))
        return [row["memory_id"] for row in cursor.fetchall()]
    elif direction == "incoming":
        cursor.execute("""
            SELECT source_memory_id AS memory_id FROM memory_relations
            WHERE target_memory_id = ?
        """, (memory_id,))
        return [row["memory_id"] for row in cursor.fetchall()]
    elif direction == "both":
        cursor.execute("""
            SELECT target_memory_id AS memory_id FROM memory_relations WHERE source_memory_id = ?
            UNION
            SELECT source_memory_id AS memory_id FROM memory_relations WHERE target_memory_id = ?
        """, (memory_id, memory_id))
        return [row["memory_id"] for row in cursor.fetchall()]
    else:
        raise ValueError(f"Unknown direction '{direction}'. Must be 'outgoing', 'incoming', or 'both'.")


def detect_chain_type(source_affect: str, target_affect: str) -> str:
    """Return 'contradicts' if affects are opposite polarity, 'confirms' otherwise."""
    src_pos = source_affect in _POSITIVE_AFFECTS
    src_neg = source_affect in _NEGATIVE_AFFECTS
    tgt_pos = target_affect in _POSITIVE_AFFECTS
    tgt_neg = target_affect in _NEGATIVE_AFFECTS

    if (src_pos and tgt_neg) or (src_neg and tgt_pos):
        return "contradicts"
    return "confirms"
