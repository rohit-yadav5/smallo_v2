"""memory_system/embeddings/eviction.py — Memory cap + FAISS LRU eviction.

When the total memory count exceeds MAX_MEMORIES, the lowest-importance
memories are deleted from SQLite (CASCADE removes their embeddings and
entity links) and the FAISS index is rebuilt from the surviving records.

Rebuilding IndexFlatIP is the only safe way to "delete" vectors because
FAISS FlatIP has no remove_ids() operation.  At <= 1000 vectors the rebuild
takes ~100-200 ms and is negligible compared to the memory write pipeline.

Configuration
─────────────
  MEMORY_MAX_COUNT  env var (default 1000)
  Protected types   PersonalMemory, ConsolidatedMemory — never evicted
                    (they hold permanent user profile and summaries)

Thread safety
─────────────
Called from the synchronous insert pipeline thread (not the event loop).
replace_index() uses the GIL to atomically swap the global FAISS reference.
"""

import logging
import os

import faiss
import numpy as np

_log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_MEMORIES: int = int(os.getenv("MEMORY_MAX_COUNT", "1000"))

# Memory types that are never evicted (too important to lose automatically)
_PROTECTED_TYPES = {"PersonalMemory", "ConsolidatedMemory"}


def get_memory_count() -> int:
    """Return total non-archived memory count from SQLite."""
    from memory_system.db.connection import get_connection  # noqa: PLC0415
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM memories")
    cnt = int(cur.fetchone()["cnt"])
    conn.close()
    return cnt


def evict_and_rebuild() -> int:
    """
    Evict the lowest-importance non-protected memories until count <= MAX_MEMORIES,
    then rebuild the FAISS index from the surviving records.

    Returns the number of memories evicted (0 if already within cap).
    Non-fatal — logs errors but never raises.
    """
    from memory_system.db.connection import get_connection          # noqa: PLC0415
    from memory_system.embeddings.embedder import generate_embedding_vector  # noqa: PLC0415
    from memory_system.embeddings.vector_store import (            # noqa: PLC0415
        VECTOR_DIM, INDEX_PATH, replace_index,
    )

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as cnt FROM memories")
    total = int(cur.fetchone()["cnt"])

    if total <= MAX_MEMORIES:
        conn.close()
        return 0

    n_evict = total - MAX_MEMORIES + 1   # +1 to make room for the next insert

    # ── Find eviction candidates ────────────────────────────────────────────
    protected_ph = ",".join("?" * len(_PROTECTED_TYPES))
    cur.execute(
        f"""
        SELECT id FROM memories
        WHERE memory_type NOT IN ({protected_ph})
        ORDER BY importance_score ASC, created_at ASC
        LIMIT ?
        """,
        (*_PROTECTED_TYPES, n_evict),
    )
    to_delete = [r["id"] for r in cur.fetchall()]

    if not to_delete:
        # Nothing evictable (everything is protected) — nothing we can do
        _log.warning("eviction: all %d memories are protected, skipping", total)
        conn.close()
        return 0

    # ── Delete from SQLite (CASCADE removes memory_embeddings + memory_entities)
    placeholders = ",".join("?" * len(to_delete))
    cur.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", to_delete)
    conn.commit()

    print(
        f"  [memory] evicted {len(to_delete)} memories  (cap={MAX_MEMORIES})",
        flush=True,
    )

    # ── Fetch surviving memories that have embeddings ───────────────────────
    cur.execute(
        """
        SELECT m.id, m.raw_text
        FROM memories m
        JOIN memory_embeddings me ON m.id = me.memory_id
        ORDER BY m.created_at ASC
        """
    )
    remaining = cur.fetchall()
    conn.close()

    if not remaining:
        # No survivors with embeddings — create an empty index
        new_index = faiss.IndexFlatIP(VECTOR_DIM)
        replace_index(new_index)
        faiss.write_index(new_index, str(INDEX_PATH))
        _log.info("FAISS rebuilt (empty — no memories with embeddings)")
        return len(to_delete)

    # ── Re-encode surviving memories ────────────────────────────────────────
    memory_ids: list[str] = []
    vectors: list[np.ndarray] = []

    for row in remaining:
        text = row["raw_text"] or ""
        if not text.strip():
            continue
        try:
            vec = generate_embedding_vector(text)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
            memory_ids.append(row["id"])
        except Exception as exc:
            _log.warning("eviction: failed to encode memory %s: %s", row["id"], exc)

    if not vectors:
        new_index = faiss.IndexFlatIP(VECTOR_DIM)
        replace_index(new_index)
        faiss.write_index(new_index, str(INDEX_PATH))
        return len(to_delete)

    # ── Build new index atomically ─────────────────────────────────────────
    matrix = np.array(vectors, dtype=np.float32)
    new_index = faiss.IndexFlatIP(VECTOR_DIM)
    new_index.add(matrix)

    # Swap the global index reference before writing to disk so all subsequent
    # searches and adds use the new index immediately.
    replace_index(new_index)
    faiss.write_index(new_index, str(INDEX_PATH))

    # ── Update memory_embeddings: assign new sequential vector_ids ─────────
    conn2 = get_connection()
    cur2 = conn2.cursor()
    for i, memory_id in enumerate(memory_ids):
        cur2.execute(
            "UPDATE memory_embeddings SET vector_id = ? WHERE memory_id = ?",
            (str(i), memory_id),
        )
    conn2.commit()
    conn2.close()

    print(
        f"  [memory] FAISS rebuilt  {new_index.ntotal} vectors  "
        f"(evicted {len(to_delete)})",
        flush=True,
    )
    return len(to_delete)
