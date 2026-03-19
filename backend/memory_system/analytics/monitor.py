from collections import Counter
from memory_system.db.connection import get_connection


def generate_health_report(verbose: bool = True):

    conn = get_connection()
    cursor = conn.cursor()

    report = {}

    # -----------------------------
    # Memory Count
    # -----------------------------
    cursor.execute("SELECT COUNT(*) as count FROM memories")
    report["total_memories"] = cursor.fetchone()["count"]

    # -----------------------------
    # Memory Type Distribution
    # -----------------------------
    cursor.execute("""
        SELECT memory_type, COUNT(*) as count
        FROM memories
        GROUP BY memory_type
    """)
    types = cursor.fetchall()
    report["memory_type_distribution"] = {
        row["memory_type"]: row["count"] for row in types
    }

    # -----------------------------
    # Lifecycle Distribution
    # -----------------------------
    cursor.execute("""
        SELECT stage, COUNT(*) as count
        FROM memory_lifecycle
        GROUP BY stage
    """)
    lifecycle = cursor.fetchall()
    report["lifecycle_distribution"] = {
        row["stage"]: row["count"] for row in lifecycle
    }

    # -----------------------------
    # Average Importance
    # -----------------------------
    cursor.execute("SELECT AVG(importance_score) as avg_importance FROM memories")
    avg_imp = cursor.fetchone()["avg_importance"]
    report["average_importance"] = round(avg_imp or 0, 3)

    # -----------------------------
    # High Importance Drift
    # -----------------------------
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM memories
        WHERE importance_score >= 8
    """)
    report["high_importance_count"] = cursor.fetchone()["count"]

    # -----------------------------
    # Entity Count
    # -----------------------------
    cursor.execute("SELECT COUNT(*) as count FROM entities")
    report["total_entities"] = cursor.fetchone()["count"]

    # -----------------------------
    # Top Entities by Usage
    # -----------------------------
    cursor.execute("""
        SELECT name, usage_count
        FROM entities
        ORDER BY usage_count DESC
        LIMIT 5
    """)
    top_entities = cursor.fetchall()
    report["top_entities"] = [
        {"name": row["name"], "usage_count": row["usage_count"]}
        for row in top_entities
    ]

    # -----------------------------
    # Duplicate Summary Detection
    # -----------------------------
    cursor.execute("SELECT summary FROM memories")
    summaries = [row["summary"] for row in cursor.fetchall()]
    duplicates = [
        item for item, count in Counter(summaries).items() if count > 1
    ]
    report["duplicate_summaries"] = len(duplicates)

    conn.close()

    if verbose:
        print("\n===== MEMORY HEALTH REPORT =====\n")
        for key, value in report.items():
            print(f"{key}: {value}")
        print("\n================================\n")

    return report