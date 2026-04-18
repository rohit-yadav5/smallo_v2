"""backend/memory_system/db/migrate_session.py — Session-ID column migration.

Adds session_id TEXT DEFAULT 'legacy' to the memories table if it does not
already exist, then marks all pre-existing rows as 'legacy'.  Running this
more than once is safe (idempotent).

Called from backend/main.py at startup before serving any requests so that
retrieval scoring can apply the cross-session penalty (FIX2A — BUG-005).
"""

from .connection import get_connection


def migrate_add_session_id() -> None:
    """
    Idempotent startup migration — adds session_id column to memories.

    Steps
    -----
    1. Check if session_id column already exists (PRAGMA table_info).
    2. If not: ALTER TABLE to add it with DEFAULT 'legacy'.
    3. Backfill any rows that have NULL or empty session_id → 'legacy'.
       This covers rows inserted before this migration ran.

    Safe to call multiple times; no-ops on step 2 if column already exists.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Step 1: check current schema
        cursor.execute("PRAGMA table_info(memories)")
        columns = [row["name"] for row in cursor.fetchall()]

        if "session_id" not in columns:
            conn.execute(
                "ALTER TABLE memories ADD COLUMN session_id TEXT DEFAULT 'legacy'"
            )
            conn.commit()
            print("  [memory] migration: added session_id column to memories", flush=True)
        else:
            print("  [memory] migration: session_id column already exists — skipping", flush=True)

        # Step 2: backfill any rows with NULL/empty session_id → 'legacy'
        cursor.execute(
            "UPDATE memories SET session_id = 'legacy' "
            "WHERE session_id IS NULL OR session_id = ''"
        )
        backfilled = cursor.rowcount
        conn.commit()
        if backfilled:
            print(
                f"  [memory] migration: marked {backfilled} legacy memories (session_id='legacy')",
                flush=True,
            )

    except Exception as exc:
        conn.rollback()
        print(f"  [memory] migration error (non-fatal): {exc}", flush=True)
    finally:
        conn.close()
