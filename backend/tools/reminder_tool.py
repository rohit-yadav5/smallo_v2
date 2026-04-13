"""backend/tools/reminder_tool.py — Reminder / timer tool for Small O.

Registers three tools at import time:
  • set_reminder    — schedule an asyncio task that fires a PROACTIVE_EVENT
                      WebSocket message after delay_seconds.
  • list_reminders  — list all pending reminders with remaining time.
  • cancel_reminder — cancel a pending reminder by its ID.

Also exposes two internal helpers (not registered as tools):
  • set_broadcast_fn(fn)      — called by main.py to inject the emit callback.
  • shutdown_all_reminders()  — awaitable; cancels every pending task cleanly.
                                Called by main.py during server shutdown.

Broadcast coupling
──────────────────
The reminder needs to push a message to all connected WebSocket clients.
Rather than importing backend/main.py (circular import), we use a
module-level setter pattern: main.py calls ``set_broadcast_fn(fn)`` once
during startup.  The fn signature matches _emit in main.py:

    broadcast_fn(event: str, data: dict) -> None

Phase 2 note: PROACTIVE_EVENT is the generic channel for autonomous agent
messages.  Future agents (task planner, web agent) will also use this event
type with different ``event`` sub-keys.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from tools.registry import ToolDefinition, registry


# ── Broadcast bridge (set by main.py at startup) ─────────────────────────────

_broadcast_fn: Callable | None = None


def set_broadcast_fn(fn: Callable) -> None:
    """Called once by backend/main.py after its _emit function is defined."""
    global _broadcast_fn
    _broadcast_fn = fn
    print("  [tools/reminder] broadcast function registered", flush=True)


# ── In-memory reminder store ──────────────────────────────────────────────────

@dataclass
class _Reminder:
    id: int
    message: str
    fire_at: float        # monotonic time
    task: asyncio.Task    # the asyncio task driving the countdown


# All pending reminders keyed by ID.
# Tasks remove themselves from here when they fire; cancel_reminder removes
# them early; shutdown_all_reminders drains the whole dict.
_reminders: dict[int, _Reminder] = {}
_next_id: int = 1


def _remove_reminder(reminder_id: int) -> None:
    _reminders.pop(reminder_id, None)


async def _reminder_task(reminder_id: int, message: str, delay_s: int) -> None:
    """Background coroutine: sleep, fire PROACTIVE_EVENT, remove self."""
    try:
        await asyncio.sleep(delay_s)
    except asyncio.CancelledError:
        # Task was cancelled (server shutdown or cancel_reminder) — exit silently.
        return
    finally:
        # Always remove from the registry so list_reminders stays accurate.
        _remove_reminder(reminder_id)

    if _broadcast_fn is not None:
        _broadcast_fn("PROACTIVE_EVENT", {
            "event":   "reminder",
            "message": message,
        })
        print(f"  [reminder] fired: '{message}'", flush=True)
    else:
        print(f"  [reminder] ⚠ no broadcast_fn set — reminder '{message}' lost", flush=True)


# ── Internal shutdown helper ──────────────────────────────────────────────────

async def shutdown_all_reminders() -> None:
    """Cancel every pending reminder task.  Called during server shutdown.

    This is NOT registered as a tool — it is an internal lifecycle hook.
    """
    ids = list(_reminders.keys())
    for rid in ids:
        r = _reminders.get(rid)
        if r and not r.task.done():
            r.task.cancel()
            try:
                await r.task
            except asyncio.CancelledError:
                pass
    _reminders.clear()
    if ids:
        print(f"  [reminder] shutdown: cancelled {len(ids)} pending reminder(s)", flush=True)


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def _set_reminder(args: dict) -> str:
    global _next_id

    message: str = args.get("message", "").strip()
    delay_s: int = int(args.get("delay_seconds", 60))

    if not message:
        return "Error: 'message' argument is required."
    if delay_s < 1:
        delay_s = 1

    reminder_id = _next_id
    _next_id   += 1

    fire_at = time.monotonic() + delay_s

    # Schedule on the main event loop (set by backend_loop_ref).
    task = asyncio.ensure_future(_reminder_task(reminder_id, message, delay_s))

    _reminders[reminder_id] = _Reminder(
        id=reminder_id,
        message=message,
        fire_at=fire_at,
        task=task,
    )

    minutes, seconds = divmod(delay_s, 60)
    human = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return f"Reminder set: '{message}' in {human}. (ID {reminder_id})"


async def _list_reminders(args: dict) -> str:
    if not _reminders:
        return "No pending reminders."

    now = time.monotonic()
    lines = []
    for r in sorted(_reminders.values(), key=lambda x: x.fire_at):
        remaining = max(0.0, r.fire_at - now)
        mins, secs = divmod(int(remaining), 60)
        human = f"{mins}m {secs}s" if mins else f"{secs}s"
        lines.append(f"• [{r.id}] '{r.message}' — fires in {human}")

    return "\n".join(lines)


async def _cancel_reminder(args: dict) -> str:
    reminder_id: int = int(args.get("reminder_id", -1))

    r = _reminders.get(reminder_id)
    if r is None:
        return f"No reminder with ID {reminder_id}."

    if not r.task.done():
        r.task.cancel()
        # Don't await here — the task's finally block will call _remove_reminder.

    _remove_reminder(reminder_id)
    return f"Reminder {reminder_id} cancelled."


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name="set_reminder",
    description=(
        "Schedule a reminder that will be sent to the user after a delay. "
        "Fires a proactive notification — use for timers, countdowns, and follow-ups."
    ),
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The reminder message to deliver to the user.",
            },
            "delay_seconds": {
                "type": "integer",
                "description": "How many seconds from now to fire the reminder.",
            },
        },
        "required": ["message", "delay_seconds"],
    },
    handler=_set_reminder,
))

registry.register(ToolDefinition(
    name="list_reminders",
    description="List all pending reminders with their remaining time.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    handler=_list_reminders,
))

registry.register(ToolDefinition(
    name="cancel_reminder",
    description="Cancel a pending reminder by its numeric ID (shown by list_reminders).",
    parameters={
        "type": "object",
        "properties": {
            "reminder_id": {
                "type": "integer",
                "description": "The numeric ID of the reminder to cancel.",
            },
        },
        "required": ["reminder_id"],
    },
    handler=_cancel_reminder,
))
