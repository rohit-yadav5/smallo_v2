"""backend/user_context.py — Persistent user model for Small O.

Maintains a growing JSON snapshot of who the user is.  The orchestrator loads
this at startup and injects a formatted excerpt into every LLM system prompt
so Small O always "knows" the user without relying solely on the vector memory.

The two systems are complementary:
  • user_context.json — structured facts (name, goals, preferences).
    Updated by the orchestrator when it hears new identity signals.
    Fast to read; injected as a compact string into the system prompt.
  • memory_system (FAISS + SQLite) — episodic / semantic memory.
    Semantic search retrieves relevant past context per utterance.

Storage: backend/data/user_context.json

Schema
──────
{
  "name": null | str,
  "preferences": { key: value, ... },
  "goals": [ str, ... ],
  "facts": [ str, ... ],
  "last_updated": "ISO-8601 timestamp"
}

Phase 2 note: when the task-planner is added, it will write structured goals
here after completing multi-step tasks, giving the next session awareness of
outstanding work.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


_DATA_DIR = Path(__file__).resolve().parent / "data"
_CTX_FILE = _DATA_DIR / "user_context.json"

# In-memory cache — avoids disk I/O on every LLM call.
_cache: dict = {}
_lock: threading.Lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_context() -> dict:
    return {
        "name":         None,
        "preferences":  {},
        "goals":        [],
        "facts":        [],
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def load_user_context() -> dict:
    """Load context from disk into the in-memory cache.

    Returns the loaded dict (also available via get_context_prompt afterwards).
    If the file doesn't exist, initialises an empty context and saves it.
    """
    global _cache
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _CTX_FILE.exists():
        try:
            raw = _CTX_FILE.read_text(encoding="utf-8")
            ctx = json.loads(raw)
            # Back-fill any keys added in future schema versions
            for k, v in _default_context().items():
                ctx.setdefault(k, v)
            with _lock:
                _cache = ctx
            print(f"  [user_ctx] loaded from {_CTX_FILE}", flush=True)
            return ctx
        except Exception as exc:
            print(f"  [user_ctx] ⚠ failed to load ({exc}) — using empty context", flush=True)

    ctx = _default_context()
    with _lock:
        _cache = ctx
    save_user_context(ctx)
    return ctx


def save_user_context(ctx: dict) -> None:
    """Persist the context dict to disk, updating last_updated timestamp."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    ctx["last_updated"] = datetime.now(timezone.utc).isoformat()
    with _lock:
        _cache.update(ctx)
    try:
        _CTX_FILE.write_text(
            json.dumps(ctx, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"  [user_ctx] ⚠ failed to save: {exc}", flush=True)


def update_user_context(key: str, value) -> None:
    """Update a single top-level key in the context and persist."""
    with _lock:
        current = dict(_cache)
    current[key] = value
    save_user_context(current)
    print(f"  [user_ctx] updated '{key}'", flush=True)


def get_user_context() -> dict:
    """Return the current in-memory context dict (no disk I/O)."""
    with _lock:
        return dict(_cache)


def get_context_prompt() -> str:
    """Return a compact formatted string for injection into the LLM system prompt."""
    with _lock:
        ctx = dict(_cache)

    name  = ctx.get("name") or "unknown"
    goals = ctx.get("goals", [])
    prefs = ctx.get("preferences", {})
    facts = ctx.get("facts", [])

    goals_str = ", ".join(goals) if goals else "none yet"
    prefs_str = (
        ";  ".join(f"{k}: {v}" for k, v in prefs.items())
        if prefs else "none yet"
    )
    facts_str = ";  ".join(facts) if facts else "none yet"

    return (
        "## What I know about you\n"
        f"Name: {name}\n"
        f"Goals: {goals_str}\n"
        f"Preferences: {prefs_str}\n"
        f"Facts: {facts_str}"
    )
