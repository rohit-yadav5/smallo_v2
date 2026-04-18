"""Async LLM summary generation module with importance bumping."""

import asyncio
from memory_system.db.connection import get_connection

_HIGH_SIGNAL_PHRASES = [
    "i prefer", "i want", "i need", "i always", "i never",
    "my goal", "decided to", "we decided", "the decision",
    "chosen to", "important to me", "make sure",
    "never do", "always do", "my preference",
]


def _do_db_write(summary: str, bump: float, memory_id: str) -> None:
    """Blocking DB write — run via asyncio.to_thread, never call from event loop directly."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE memories
            SET summary = ?,
                importance_score = MIN(importance_score + ?, 10.0)
            WHERE id = ?
        """, (summary, bump, memory_id))
        conn.commit()
    finally:
        conn.close()


async def generate_and_store_summary(memory_id: str, raw_text: str, memory_type: str) -> None:
    """Fire-and-forget async task: generates LLM summary and updates the DB record.

    The memory record is already inserted with a truncated placeholder before this
    task runs — so retrieval always returns *something* even if this hasn't finished.
    """
    try:
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(
            None,
            lambda: _call_llm_for_summary(raw_text, memory_type),
        )
        bump = _compute_importance_bump(summary)

        await asyncio.to_thread(_do_db_write, summary, bump, memory_id)
    except Exception as exc:
        print(f"[memory] async summary failed for {memory_id}: {exc}", flush=True)


def _call_llm_for_summary(raw_text: str, memory_type: str) -> str:
    """Blocking LLM call — run via executor, never call from event loop directly."""
    try:
        from llm import ask_llm
        prompt = (
            f"Summarize this memory in one concise sentence. Preserve all key facts.\n\n"
            f"Memory type: {memory_type}\n"
            f"Text: {raw_text[:500]}\n\n"
            "One sentence summary:"
        )
        result = ask_llm(prompt).strip()
        return result[:300] if result else raw_text[:300]
    except Exception:
        return raw_text[:300]


def _compute_importance_bump(summary: str) -> float:
    """Return +2.0 if summary contains a high-signal phrase, else 0.0."""
    lower = summary.lower()
    for phrase in _HIGH_SIGNAL_PHRASES:
        if phrase in lower:
            return 2.0
    return 0.0
