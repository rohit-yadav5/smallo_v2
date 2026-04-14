import uuid
from datetime import datetime

from memory_system.db.connection import get_connection
from memory_system.core.importance import calculate_importance
from memory_system.entities.extractor import extract_entities
from memory_system.entities.service import get_or_create_entity
from memory_system.embeddings.embedder import generate_embedding_vector
from memory_system.embeddings.vector_store import add_vector
from memory_system.embeddings.vector_store import search_vector


def insert_memory(input_data: dict):

    memory_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    raw_text = input_data["text"]
    source = input_data.get("source", "manual")

    # Step 1: classify type (temporary default)
    memory_type = input_data.get("memory_type", "IdeaMemory")

    # Step 2: summary (temporary = raw_text)
    summary = raw_text[:300]

    # -----------------------------
    # 🔎 Pre-insert Deduplication
    # -----------------------------
    DEDUP_THRESHOLD = 0.90

    embedding_input = f"{memory_type} | {summary}"
    new_vector = generate_embedding_vector(embedding_input)
    distances, indices = search_vector(new_vector, top_k=3)

    conn = get_connection()
    cursor = conn.cursor()

    for distance, idx in zip(distances, indices):

        if idx == -1:
            continue

        # Using inner product (cosine similarity) from FAISS IndexFlatIP
        similarity = float(distance)

        if similarity >= DEDUP_THRESHOLD:

            cursor.execute("""
                SELECT memory_id FROM memory_embeddings
                WHERE vector_id = ?
            """, (str(idx),))

            row = cursor.fetchone()

            if row:
                existing_memory_id = row["memory_id"]
                print(f"[Dedup] Skipping insert. Similar memory exists: {existing_memory_id}")
                conn.close()
                return existing_memory_id

    conn.close()

    # Step 3: extract entities
    entities = extract_entities(raw_text)

    # Step 4: importance score
    importance = calculate_importance(memory_type, raw_text)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Insert memory
        cursor.execute("""
            INSERT INTO memories
            (id, memory_type, raw_text, summary, importance_score, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            memory_type,
            raw_text,
            summary,
            importance,
            source,
            created_at
        ))

        # ENTITY LINKING (transaction-safe)
        for entity in entities:
            entity_id = get_or_create_entity(
                cursor,
                name=entity["name"],
                domain=entity["domain"],
                category=entity["category"],
                entity_type=entity["entity_type"]
            )

            cursor.execute("""
                INSERT OR IGNORE INTO memory_entities (memory_id, entity_id)
                VALUES (?, ?)
            """, (memory_id, entity_id))

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

    # Step 5: reuse precomputed embedding vector
    vector = new_vector

    # Step 6: store in FAISS using stable memory_id mapping (IDMap)
    numeric_id = add_vector(memory_id, vector)

    # Step 7: store embedding reference in SQL
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO memory_embeddings (memory_id, vector_id, model_name)
        VALUES (?, ?, ?)
    """, (
        memory_id,
        str(numeric_id),
        "all-MiniLM-L6-v2"
    ))

    conn.commit()
    conn.close()

    # ── Memory cap check ────────────────────────────────────────────────────
    # After every insert, check if we've exceeded MAX_MEMORIES.
    # If so, evict lowest-importance memories and rebuild FAISS.
    # This keeps retrieval latency bounded and RAM usage predictable.
    try:
        from memory_system.embeddings.eviction import (  # noqa: PLC0415
            MAX_MEMORIES, get_memory_count, evict_and_rebuild,
        )
        current_count = get_memory_count()
        if current_count > MAX_MEMORIES:
            print(
                f"  [memory] cap exceeded ({current_count}/{MAX_MEMORIES}) — evicting",
                flush=True,
            )
            evict_and_rebuild()
    except Exception as _evict_exc:
        print(f"  [memory] eviction check failed (non-fatal): {_evict_exc}", flush=True)

    return memory_id