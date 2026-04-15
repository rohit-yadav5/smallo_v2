"""backend/tools/memory_tool.py — Memory management tool for Small O.

Provides the clear_memory tool which wipes ALL memories from SQLite + FAISS.
Requires confirm=True to prevent accidental data loss.
"""

import faiss
from tools.registry import ToolDefinition, registry


def reset_all_memories() -> str:
    """
    Wipe every memory from the SQLite database and reset the FAISS index.
    Returns a human-readable result string.
    """
    deleted_count = 0
    try:
        from memory_system.db.connection import get_connection, DB_PATH  # noqa: PLC0415
        conn = get_connection()
        cursor = conn.cursor()
        # Count before deletion for the summary message
        cursor.execute("SELECT COUNT(*) FROM memories")
        row = cursor.fetchone()
        deleted_count = row[0] if row else 0
        # Wipe all tables that hold memory data
        cursor.execute("DELETE FROM memory_embeddings")
        cursor.execute("DELETE FROM memory_entities")
        cursor.execute("DELETE FROM memories")
        conn.commit()
        conn.close()
    except Exception as e:
        return f"Error clearing SQLite memories: {e}"

    try:
        from memory_system.embeddings.vector_store import (  # noqa: PLC0415
            VECTOR_DIM, INDEX_PATH, replace_index,
        )
        new_index = faiss.IndexFlatIP(VECTOR_DIM)
        faiss.write_index(new_index, str(INDEX_PATH))
        replace_index(new_index)
    except Exception as e:
        return f"SQLite cleared ({deleted_count} memories) but FAISS reset failed: {e}"

    return f"Cleared {deleted_count} memories from storage. Fresh start."


async def _clear_memory(args: dict) -> str:
    confirm = args.get("confirm", False)
    if not confirm:
        return (
            "Safety check: clear_memory requires confirm=true. "
            "This will permanently delete ALL stored memories."
        )
    return reset_all_memories()


registry.register(ToolDefinition(
    name="clear_memory",
    description=(
        "Permanently delete ALL stored memories. "
        "Use ONLY when the user explicitly asks to clear, reset, or wipe memory. "
        "Requires confirm=true to prevent accidental data loss."
    ),
    parameters={
        "type": "object",
        "properties": {
            "confirm": {
                "type": "boolean",
                "description": "Must be true to execute. Prevents accidental deletion.",
            },
        },
        "required": ["confirm"],
    },
    handler=_clear_memory,
))
