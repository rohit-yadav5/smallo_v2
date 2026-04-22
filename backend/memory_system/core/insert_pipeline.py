import asyncio
import uuid
from datetime import datetime

from config.limits import DEDUP_THRESHOLD, NEAR_DEDUP_THRESHOLD
from memory_system.db.connection import get_connection, managed_connection
from memory_system.core.importance import calculate_importance
from memory_system.core.affect import detect_affect
from memory_system.core.chain import create_chain, detect_chain_type
from memory_system.entities.extractor import extract_entities
from memory_system.entities.service import get_or_create_entity
from memory_system.embeddings.embedder import generate_embedding_vector
from memory_system.embeddings.vector_store import add_vector
from memory_system.embeddings.vector_store import search_vector


# Holds in-flight summary futures so they aren't GC'd before completion
_pending_summary_futures: set = set()


def _get_session_id() -> str:
    try:
        import backend_loop_ref as _ref
        sid = getattr(_ref, "session_id", None)
        if not sid:
            raise RuntimeError(
                "backend_loop_ref.session_id is not set — "
                "insert_memory called before backend initialisation completed"
            )
        return sid
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not read session_id: {exc}") from exc


def insert_memory(input_data: dict) -> str:
    memory_id   = str(uuid.uuid4())
    created_at  = datetime.utcnow().isoformat()
    raw_text    = input_data["text"]
    source      = input_data.get("source", "manual")
    memory_type = input_data.get("memory_type", "IdeaMemory")
    summary     = raw_text[:300]
    session_id  = _get_session_id()

    # ── Step 1: Affect tagging ────────────────────────────────────────────────
    affect = detect_affect(raw_text)

    # ── Step 2: Pre-insert deduplication + near-duplicate chain detection ─────
    embedding_input = f"{memory_type} | {summary}"
    new_vector = generate_embedding_vector(embedding_input)
    distances, indices = search_vector(new_vector, top_k=5)

    near_duplicates: list[tuple[str, str]] = []   # (existing_memory_id, existing_affect)

    with managed_connection() as conn:
        cursor = conn.cursor()

        for distance, idx in zip(distances, indices):
            if idx == -1:
                continue
            similarity = float(distance)

            if similarity >= DEDUP_THRESHOLD:
                cursor.execute("""
                    SELECT memory_id FROM memory_embeddings WHERE vector_id = ?
                """, (str(idx),))
                row = cursor.fetchone()
                if row:
                    existing_id = row["memory_id"]
                    if session_id not in ("legacy",):
                        cursor.execute(
                            "UPDATE memories SET session_id = ? WHERE id = ?",
                            (session_id, existing_id),
                        )
                        conn.commit()
                    return existing_id

            elif similarity >= NEAR_DEDUP_THRESHOLD:
                cursor.execute("""
                    SELECT me.memory_id, m.affect
                    FROM memory_embeddings me
                    JOIN memories m ON m.id = me.memory_id
                    WHERE me.vector_id = ?
                """, (str(idx),))
                row = cursor.fetchone()
                if row:
                    near_duplicates.append((row["memory_id"], row["affect"] or "neutral"))

    # ── Step 3: Entity extraction + importance ────────────────────────────────
    entities   = extract_entities(raw_text)
    importance = calculate_importance(memory_type, raw_text)

    # ── Step 4: Insert into SQLite ────────────────────────────────────────────
    with managed_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO memories
            (id, memory_type, raw_text, summary, importance_score, source,
             session_id, affect, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, memory_type, raw_text, summary,
            importance, source, session_id, affect, created_at,
        ))

        for entity in entities:
            entity_id = get_or_create_entity(
                cursor,
                name=entity["name"],
                domain=entity["domain"],
                category=entity["category"],
                entity_type=entity["entity_type"],
            )
            cursor.execute("""
                INSERT OR IGNORE INTO memory_entities (memory_id, entity_id)
                VALUES (?, ?)
            """, (memory_id, entity_id))

        # ── Step 4b: Chain links for near-duplicates ──────────────────────────
        for existing_id, existing_affect in near_duplicates:
            chain_type = detect_chain_type(affect, existing_affect)
            create_chain(cursor, memory_id, existing_id, chain_type)
            if chain_type == "contradicts":
                cursor.execute("""
                    UPDATE memories
                    SET confidence_score = MAX(COALESCE(confidence_score, 1.0) - 0.2, 0.0)
                    WHERE id = ?
                """, (existing_id,))
            elif chain_type == "confirms":
                cursor.execute("""
                    UPDATE memories
                    SET confidence_score = MIN(COALESCE(confidence_score, 1.0) + 0.1, 1.0)
                    WHERE id = ?
                """, (existing_id,))

        conn.commit()

    # ── Step 5: FAISS insert (atomic with SQLite embedding record) ───────────────
    try:
        numeric_id = add_vector(memory_id, new_vector)
    except Exception as exc:
        # FAISS insert failed — delete the SQLite row so we don't have orphaned memory
        print(f"[memory] FAISS insert failed for {memory_id} — rolling back SQLite row: {exc}", flush=True)
        with managed_connection() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
        return ""

    with managed_connection() as conn:
        conn.execute("""
            INSERT INTO memory_embeddings (memory_id, vector_id, model_name)
            VALUES (?, ?, ?)
        """, (memory_id, str(numeric_id), "all-MiniLM-L6-v2"))

    # ── Step 6: Async summary generation (fire-and-forget) ───────────────────
    try:
        from memory_system.core.async_summary import generate_and_store_summary
        import backend_loop_ref as _blr
        loop = getattr(_blr, "loop", None)
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                generate_and_store_summary(memory_id, raw_text, memory_type),
                loop,
            )
            # Store reference so GC doesn't collect it before it runs
            _pending_summary_futures.add(future)
            future.add_done_callback(_pending_summary_futures.discard)
    except Exception as exc:
        print(f"[memory] async_summary unavailable: {exc}", flush=True)

    # ── Step 7: Memory cap check ──────────────────────────────────────────────
    try:
        from memory_system.embeddings.eviction import (
            MAX_MEMORIES, get_memory_count, evict_and_rebuild,
        )
        if get_memory_count() > MAX_MEMORIES:
            evict_and_rebuild()
    except Exception as exc:
        print(f"  [memory] eviction check failed (non-fatal): {exc}", flush=True)

    return memory_id
