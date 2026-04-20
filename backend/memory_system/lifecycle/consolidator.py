import uuid
from datetime import datetime
from collections import defaultdict

from memory_system.db.connection import get_connection
from memory_system.embeddings.embedder import generate_embedding_vector
from memory_system.embeddings.vector_store import search_vector

from llm import ask_llm

from config.llm import (
    CONSOLIDATION_SIMILARITY_THRESHOLD,
    META_CONSOLIDATION_SIMILARITY_THRESHOLD,
    CONSOLIDATION_VALIDATION_THRESHOLD,
    CONSOLIDATED_MEMORY_EXPIRY_DAYS,
    CONSOLIDATED_MEMORY_EXPIRY_MIN_IMPORTANCE,
)
SIMILARITY_THRESHOLD = CONSOLIDATION_SIMILARITY_THRESHOLD
MIN_CLUSTER_SIZE = 3


def similarity_from_distance(distance):
    return 1 / (1 + float(distance))


def generate_llm_consolidation(project, summaries):
    """
    Use local LLM to generate structured consolidation.
    """

    prompt = f"""
You are consolidating weekly project work.

Project: {project}

Below are related memory summaries:
{chr(10).join("- " + s for s in summaries)}

Generate:
1. A concise title
2. A structured summary paragraph
3. 3-5 key themes

Return in this format:

TITLE:
...

SUMMARY:
...

KEY THEMES:
- ...
- ...
"""

    response = ask_llm(prompt)
    return response.strip()


def validate_consolidation(summary: str, source_summaries: list[str]) -> bool:
    """Return True if the LLM summary is semantically close to the cluster centroid."""
    import numpy as np
    summary_vec = generate_embedding_vector(summary)
    source_vecs = np.array([generate_embedding_vector(s) for s in source_summaries])
    centroid    = source_vecs.mean(axis=0)
    norm_s = np.linalg.norm(summary_vec)
    norm_c = np.linalg.norm(centroid)
    if norm_s < 1e-8 or norm_c < 1e-8:
        return False
    similarity = float(np.dot(summary_vec, centroid) / (norm_s * norm_c))
    return similarity >= CONSOLIDATION_VALIDATION_THRESHOLD


def run_weekly_consolidation():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT m.id, m.summary, m.project_reference
        FROM memories m
        LEFT JOIN memory_lifecycle l ON m.id = l.memory_id
        WHERE (l.stage IS NULL OR l.stage = 'raw')
    """)

    rows = cursor.fetchall()

    project_groups = defaultdict(list)

    for row in rows:
        project = row["project_reference"] or "global"
        project_groups[project].append(row)

    consolidated_count = 0
    # Track globally consolidated in this run to prevent overlap
    globally_used_memory_ids = set()

    try:
        for project, memories in project_groups.items():

            # Track per-project usage but respect global usage
            used_memory_ids = set()

            for memory in memories:

                # Skip if already clustered in this run
                if memory["id"] in used_memory_ids or memory["id"] in globally_used_memory_ids:
                    continue

                query_vector = generate_embedding_vector(memory["summary"])
                distances, indices = search_vector(query_vector, top_k=5)

                cluster = [memory]
                used_memory_ids.add(memory["id"])

                for distance, idx in zip(distances, indices):

                    if idx == -1:
                        continue

                    similarity = similarity_from_distance(distance)

                    if similarity < SIMILARITY_THRESHOLD:
                        continue

                    cursor.execute("""
                        SELECT memory_id FROM memory_embeddings
                        WHERE vector_id = ?
                    """, (str(idx),))

                    result = cursor.fetchone()
                    if not result:
                        continue

                    candidate_id = result["memory_id"]

                    if candidate_id == memory["id"]:
                        continue

                    cursor.execute("""
                        SELECT id, summary, project_reference
                        FROM memories
                        WHERE id = ?
                    """, (candidate_id,))

                    candidate = cursor.fetchone()
                    if not candidate:
                        continue

                    if (candidate["project_reference"] or "global") != project:
                        continue

                    if (
                        candidate_id not in used_memory_ids
                        and candidate_id not in globally_used_memory_ids
                    ):
                        cluster.append(candidate)
                        used_memory_ids.add(candidate_id)

                if len(cluster) >= MIN_CLUSTER_SIZE:

                    summaries = [m["summary"] for m in cluster]

                    # 🔥 LLM consolidation here
                    consolidated_text = generate_llm_consolidation(project, summaries)
                    if not validate_consolidation(consolidated_text, summaries):
                        consolidated_text = generate_llm_consolidation(project, summaries)
                        if not validate_consolidation(consolidated_text, summaries):
                            print(f"[Consolidation] Skipping cluster in '{project}' — validation failed twice.")
                            continue

                    new_id = f"consolidated-{datetime.utcnow().timestamp()}"

                    cursor.execute("""
                        INSERT INTO memories
                        (id, memory_type, raw_text, summary,
                         importance_score, source, created_at, project_reference)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        new_id,
                        "ConsolidatedMemory",
                        consolidated_text,
                        consolidated_text,
                        8,  # Higher importance
                        "system",
                        datetime.utcnow().isoformat(),
                        project
                    ))
                    # Initialize lifecycle record for consolidated memory
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_lifecycle
                        (memory_id, stage)
                        VALUES (?, 'raw')
                    """, (new_id,))

                    # Mark cluster memories as globally used
                    for m in cluster:
                        globally_used_memory_ids.add(m["id"])

                    # Link old memories
                    for m in cluster:
                        cursor.execute("""
                            INSERT INTO memory_relations
                            (id, source_memory_id, target_memory_id,
                             relation_type, created_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            f"rel-{m['id']}-{new_id}",
                            m["id"],
                            new_id,
                            "summarized_by",
                            datetime.utcnow().isoformat()
                        ))

                        cursor.execute("""
                            UPDATE memories
                            SET importance_score = importance_score * 0.5
                            WHERE id = ?
                        """, (m["id"],))

                        cursor.execute("""
                            INSERT OR REPLACE INTO memory_lifecycle
                            (memory_id, stage)
                            VALUES (?, 'consolidated')
                        """, (m["id"],))

                    consolidated_count += 1

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        conn.close()

    print(f"[Consolidation] Created {consolidated_count} intelligent consolidated memories.")


def run_meta_consolidation():
    """Merge pairs of ConsolidatedMemory records that are very similar (depth-2 only)."""
    import numpy as np
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, summary, project_reference FROM memories
        WHERE memory_type = 'ConsolidatedMemory' AND status != 'archived'
        LIMIT 200
    """)
    consolidated = cursor.fetchall()
    conn.close()

    used: set[str] = set()
    for i, mem_a in enumerate(consolidated):
        if mem_a["id"] in used:
            continue
        vec_a = np.array(generate_embedding_vector(mem_a["summary"]), dtype="float32")
        for mem_b in consolidated[i + 1:]:
            if mem_b["id"] in used:
                continue
            if (mem_a["project_reference"] or "global") != (mem_b["project_reference"] or "global"):
                continue
            vec_b = np.array(generate_embedding_vector(mem_b["summary"]), dtype="float32")
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                continue
            sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
            if sim >= META_CONSOLIDATION_SIMILARITY_THRESHOLD:
                summaries = [mem_a["summary"], mem_b["summary"]]
                meta_text = generate_llm_consolidation(mem_a["project_reference"] or "global", summaries)
                new_id = f"meta-{datetime.utcnow().timestamp()}"
                conn = get_connection()
                try:
                    c2 = conn.cursor()
                    c2.execute("""
                        INSERT INTO memories (id, memory_type, raw_text, summary, importance_score,
                                             source, created_at, project_reference)
                        VALUES (?, 'MetaConsolidatedMemory', ?, ?, 9, 'system', ?, ?)
                    """, (new_id, meta_text, meta_text, datetime.utcnow().isoformat(), mem_a["project_reference"]))
                    for src_id in (mem_a["id"], mem_b["id"]):
                        c2.execute("""
                            INSERT INTO memory_relations (id, source_memory_id, target_memory_id,
                                                          relation_type, created_at)
                            VALUES (?, ?, ?, 'summarized_by', ?)
                        """, (str(uuid.uuid4()), src_id, new_id, datetime.utcnow().isoformat()))
                    conn.commit()
                finally:
                    conn.close()
                used.update([mem_a["id"], mem_b["id"]])
                break


def expire_old_consolidated_memories():
    """Archive old ConsolidatedMemory records below the importance threshold."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, importance_score, created_at FROM memories
        WHERE memory_type = 'ConsolidatedMemory' AND status = 'active'
    """)
    rows = cursor.fetchall()
    conn.close()

    expired_ids = []
    for row in rows:
        try:
            age_days = (datetime.utcnow() - datetime.fromisoformat(row["created_at"])).days
        except Exception:
            age_days = 0
        if (age_days > CONSOLIDATED_MEMORY_EXPIRY_DAYS and
                row["importance_score"] < CONSOLIDATED_MEMORY_EXPIRY_MIN_IMPORTANCE):
            expired_ids.append(row["id"])

    if expired_ids:
        conn   = get_connection()
        cursor = conn.cursor()
        for mid in expired_ids:
            cursor.execute("UPDATE memories SET status = 'archived' WHERE id = ?", (mid,))
        conn.commit()
        conn.close()
        print(f"[Consolidation] Archived {len(expired_ids)} expired ConsolidatedMemory records.")