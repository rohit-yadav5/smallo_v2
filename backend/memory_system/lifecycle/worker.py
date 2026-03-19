from datetime import datetime, timedelta
from memory_system.db.connection import get_connection


ARCHIVE_DAYS_THRESHOLD = 30
LOW_IMPORTANCE_THRESHOLD = 4


def ensure_lifecycle_record(cursor, memory_id: str):
    """
    Ensure a lifecycle record exists for a memory.
    If not, create one with stage='raw'.
    """
    cursor.execute("""
        SELECT memory_id FROM memory_lifecycle
        WHERE memory_id = ?
    """, (memory_id,))
    row = cursor.fetchone()

    if not row:
        cursor.execute("""
            INSERT INTO memory_lifecycle (memory_id, stage)
            VALUES (?, 'raw')
        """, (memory_id,))


def run_lifecycle_maintenance():
    """
    Phase 1 Lifecycle:
    - Archive memories older than ARCHIVE_DAYS_THRESHOLD days
    - Only if importance_score < LOW_IMPORTANCE_THRESHOLD
    - Skip already archived memories
    """

    conn = get_connection()
    cursor = conn.cursor()

    threshold_date = (datetime.utcnow() - timedelta(days=ARCHIVE_DAYS_THRESHOLD)).isoformat()

    # Find candidate memories
    cursor.execute("""
        SELECT m.id, m.importance_score, m.created_at, l.stage
        FROM memories m
        LEFT JOIN memory_lifecycle l ON m.id = l.memory_id
        WHERE m.created_at < ?
          AND m.importance_score < ?
    """, (threshold_date, LOW_IMPORTANCE_THRESHOLD))

    candidates = cursor.fetchall()

    archived_count = 0

    try:
        for memory in candidates:

            memory_id = memory["id"]
            current_stage = memory["stage"]

            # Ensure lifecycle record exists
            ensure_lifecycle_record(cursor, memory_id)

            # Skip if already archived
            if current_stage == "archived":
                continue

            cursor.execute("""
                UPDATE memory_lifecycle
                SET stage = 'archived',
                    archived_at = ?
                WHERE memory_id = ?
            """, (
                datetime.utcnow().isoformat(),
                memory_id
            ))

            archived_count += 1

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        conn.close()

    print(f"[Lifecycle] Archived {archived_count} old low-importance memories.")