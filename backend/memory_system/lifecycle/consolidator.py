from datetime import datetime
from collections import defaultdict

from memory_system.db.connection import get_connection
from memory_system.embeddings.embedder import generate_embedding_vector
from memory_system.embeddings.vector_store import search_vector

from llm import ask_llm


SIMILARITY_THRESHOLD = 0.75
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