"""backend/backend_loop_ref.py — Shared asyncio event-loop reference.

The pipeline runs in a synchronous thread.  Tools that need to schedule
asyncio coroutines (e.g. reminder_tool's asyncio.ensure_future) must post
work to the *main* event loop rather than creating a new one.

backend/main.py sets ``backend_loop_ref.loop`` as soon as _main() starts.
backend/llm/main_llm.py reads it to route tool dispatch correctly.

Also carries ``session_id`` — the current server startup UUID — so that
memory system modules (insert_pipeline, search) can read the active session
without a circular import back to main.py.  (FIX2A — BUG-005)

This module intentionally has no imports other than the stdlib — keeping it
import-safe from any context.
"""

import asyncio

# Set by backend/main.py._main() immediately after the loop starts.
loop: asyncio.AbstractEventLoop | None = None

# Set by backend/main.py._main() after session ID is generated.
# Memory insert and retrieval modules read this to tag / filter memories.
session_id: str = "unknown"
