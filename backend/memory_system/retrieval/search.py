import concurrent.futures
from datetime import datetime
from memory_system.db.connection import get_connection
from memory_system.entities.extractor import extract_entities
from memory_system.embeddings.embedder import generate_embedding_vector
from memory_system.embeddings.vector_store import search_vector
from collections import deque

# ── Entity extraction fast-path helpers ──────────────────────────────────────
# A single-worker pool used to run entity extraction with a wall-clock timeout.
# This prevents a slow/unloaded NLP model from blocking retrieval for > 2s.
_entity_exec = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="entity_extract"
)

# Predicates that determine whether entity extraction is worth running.
# Short or trivially simple queries are handled by pure FAISS search.
_SKIP_ENTITY_PREDICATES = [
    lambda q: len(q.split()) <= 6,                                    # very short
    lambda q: q.strip().endswith("?") and len(q.split()) <= 8,       # simple question
]


def should_skip_entity_extraction(query: str) -> bool:
    """Return True if entity extraction would add no value for this query."""
    return any(fn(query) for fn in _SKIP_ENTITY_PREDICATES)


def _extract_entities_timed(query: str, timeout: float = 2.0) -> list[dict]:
    """Run extract_entities with a wall-clock timeout; returns [] on timeout."""
    future = _entity_exec.submit(extract_entities, query)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print("[memory] entity extraction timed out — using raw query", flush=True)
        return []
    except Exception as exc:
        print(f"[memory] entity extraction error — {exc}", flush=True)
        return []


def calculate_recency_boost(created_at_str):
    """Return a recency multiplier (1.0 = recent, 0.4 = old) based on memory age."""
    created_at = datetime.fromisoformat(created_at_str)
    days_old = (datetime.utcnow() - created_at).days

    if days_old < 7:
        return 1.0
    elif days_old < 30:
        return 0.7
    else:
        return 0.4


def detect_intent(query: str) -> str:
    """Classify query intent for score boosting during retrieval."""
    q = query.lower()

    # Personal memory intent
    if any(x in q for x in ["my name", "how old am i", "my friend", "who am i"]):
        return "personal_query"

    # Project recall intent
    if any(x in q for x in ["what did i", "when did i", "my project", "what have i"]):
        return "project_recall"

    # Knowledge intent
    if any(x in q for x in ["what is", "explain", "define", "how does"]):
        return "knowledge_query"

    return "general"


def retrieve_memories(query: str, top_k: int = 5, debug: bool = False):

    conn = get_connection()
    cursor = conn.cursor()

    # ---------------------------
    # 1️⃣ Extract Entities From Query (with fast-path skip + timeout)
    # ---------------------------
    if should_skip_entity_extraction(query):
        # Short / simple queries — skip entity extraction entirely.
        # FAISS semantic search on the raw text is sufficient and saves 20-2000 ms.
        query_entities = []
        entity_names: list[str] = []
        print(f"  [memory] skipped entity extraction (short query: {len(query.split())} words)", flush=True)
    else:
        # Longer queries — run entity extraction with a 2s timeout safety net.
        query_entities = _extract_entities_timed(query, timeout=2.0)
        entity_names = [e["name"] for e in query_entities]

    # ---------------------------
    # Expand entity graph (graph mode)
    # ---------------------------
    def expand_entity_graph(start_entity_ids):
        visited = set()
        queue = deque(start_entity_ids)

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue

            visited.add(current_id)

            # Find children (is_a, part_of, depends_on, etc.)
            cursor.execute("""
                SELECT target_entity_id FROM entity_relations
                WHERE source_entity_id = ?
            """, (current_id,))
            children = cursor.fetchall()

            for child in children:
                child_id = child["target_entity_id"]
                if child_id not in visited:
                    queue.append(child_id)

        return visited

    candidate_memory_ids = set()

    if entity_names:
        root_entity_ids = []

        for name in entity_names:
            cursor.execute("""
                SELECT e.id FROM entities e
                WHERE e.name = ?
            """, (name.lower(),))

            row = cursor.fetchone()
            if row:
                root_entity_ids.append(row["id"])

        # Expand graph to include related entities
        expanded_entity_ids = expand_entity_graph(root_entity_ids)

        # Collect memories linked to expanded entities
        for entity_id in expanded_entity_ids:
            cursor.execute("""
                SELECT memory_id FROM memory_entities
                WHERE entity_id = ?
            """, (entity_id,))

            rows = cursor.fetchall()
            for r in rows:
                candidate_memory_ids.add(r["memory_id"])

    # ---------------------------
    # 🎯 Detect Intent
    # ---------------------------
    intent = detect_intent(query)

    # ---------------------------
    # 2️⃣ Semantic Search
    # ---------------------------
    query_vector = generate_embedding_vector(query)
    distances, indices = search_vector(query_vector, top_k=top_k * 2)

    results = []

    for numeric_id, distance in zip(indices, distances):

        if numeric_id == -1:
            continue

        cursor.execute("""
            SELECT memory_id FROM memory_embeddings
            WHERE vector_id = ?
        """, (str(numeric_id),))

        row = cursor.fetchone()
        if not row:
            continue

        memory_id = row["memory_id"]

        # If entity filter exists, enforce it
        if candidate_memory_ids and memory_id not in candidate_memory_ids:
            continue

        cursor.execute("""
            SELECT id, summary, importance_score, created_at, memory_type
            FROM memories
            WHERE id = ?
        """, (memory_id,))

        memory = cursor.fetchone()
        if not memory:
            continue

        similarity_score = 1 / (1 + float(distance))
        importance_score = memory["importance_score"] / 10
        recency_score = calculate_recency_boost(memory["created_at"])

        # Strategic boost for consolidated memories
        consolidation_boost = 0.15 if memory["memory_type"] == "ConsolidatedMemory" else 0

        # Base score
        final_score = (
            similarity_score * 0.55 +
            importance_score * 0.2 +
            recency_score * 0.1 +
            consolidation_boost
        )

        # -----------------------------
        # 🎯 Intent-Based Adjustments
        # -----------------------------
        if intent == "personal_query":
            if memory["memory_type"] == "PersonalMemory":
                final_score += 0.2

        elif intent == "project_recall":
            if memory["memory_type"] in ["ProjectMemory", "ArchitectureMemory", "DecisionMemory", "ConsolidatedMemory"]:
                final_score += 0.15

        elif intent == "knowledge_query":
            if memory["memory_type"] == "ReflectionMemory":
                final_score -= 0.1

        explanation = {
            "similarity_score": round(similarity_score, 4),
            "importance_score": round(importance_score, 4),
            "recency_score": round(recency_score, 4),
            "consolidation_boost": consolidation_boost,
            "final_score": round(final_score, 4)
        }

        if debug:
            results.append({
                "memory_id": memory["id"],
                "summary": memory["summary"],
                "memory_type": memory["memory_type"],
                "score": final_score,
                "explanation": explanation
            })
        else:
            results.append({
                "memory_id": memory["id"],
                "summary": memory["summary"],
                "memory_type": memory["memory_type"],
                "score": final_score
            })

    # Sort by final score
    results.sort(key=lambda x: x["score"], reverse=True)

    top_results = results[:top_k]

    # -----------------------------
    # 🧠 Recall Reinforcement (Top-K Only)
    # -----------------------------
    conn = get_connection()
    cursor = conn.cursor()

    for r in top_results:
        cursor.execute("""
            SELECT importance_score FROM memories WHERE id = ?
        """, (r["memory_id"],))
        row = cursor.fetchone()
        if row:
            current_importance = row["importance_score"]
            new_importance = min(current_importance + 0.1, 10)
            cursor.execute("""
                UPDATE memories SET importance_score = ? WHERE id = ?
            """, (new_importance, r["memory_id"]))

    conn.commit()
    conn.close()

    return top_results