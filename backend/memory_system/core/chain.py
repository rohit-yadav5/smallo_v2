import uuid
from datetime import datetime

CHAIN_RELATION_TYPES = {"caused_by", "led_to", "contradicts", "confirms", "summarized_by"}

_POSITIVE_AFFECTS = {"positive", "excited"}
_NEGATIVE_AFFECTS = {"negative", "frustrated"}


def create_chain(cursor, source_id: str, target_id: str, relation_type: str) -> str:
    """Insert a directed memory chain link. Returns the relation id."""
    assert relation_type in CHAIN_RELATION_TYPES, (
        f"Unknown relation type '{relation_type}'. Must be one of {CHAIN_RELATION_TYPES}"
    )
    rel_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT OR IGNORE INTO memory_relations
        (id, source_memory_id, target_memory_id, relation_type, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (rel_id, source_id, target_id, relation_type, datetime.utcnow().isoformat()))
    return rel_id


def get_chain_links(cursor, memory_id: str, relation_type: str = None) -> list[dict]:
    """Return all memory_relations rows where source_memory_id = memory_id."""
    if relation_type:
        cursor.execute("""
            SELECT target_memory_id, relation_type FROM memory_relations
            WHERE source_memory_id = ? AND relation_type = ?
        """, (memory_id, relation_type))
    else:
        cursor.execute("""
            SELECT target_memory_id, relation_type FROM memory_relations
            WHERE source_memory_id = ?
        """, (memory_id,))
    return [dict(r) for r in cursor.fetchall()]


def detect_chain_type(source_affect: str, target_affect: str) -> str:
    """Return 'contradicts' if affects are opposite polarity, 'confirms' otherwise."""
    src_pos = source_affect in _POSITIVE_AFFECTS
    src_neg = source_affect in _NEGATIVE_AFFECTS
    tgt_pos = target_affect in _POSITIVE_AFFECTS
    tgt_neg = target_affect in _NEGATIVE_AFFECTS

    if (src_pos and tgt_neg) or (src_neg and tgt_pos):
        return "contradicts"
    return "confirms"
