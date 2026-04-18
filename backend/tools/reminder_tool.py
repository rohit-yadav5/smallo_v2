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

    # Speak the reminder aloud through the TTS pipeline.
    # Lazy import avoids a circular dependency at module load time
    # (main.py → reminder_tool → tts is fine; tts never imports reminder_tool).
    try:
        from tts import speak  # noqa: PLC0415
        await asyncio.to_thread(speak, f"Reminder: {message}", None)
    except Exception as exc:
        print(f"  [reminder] ⚠ TTS speak failed: {exc}", flush=True)


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


# ── Delay resolver ────────────────────────────────────────────────────────────

def _resolve_delay(args: dict) -> int | str:
    """
    Defensively extract the delay in seconds from LLM args.

    The canonical key is ``delay_seconds``.  The LLM occasionally hallucinates
    alternative names.  We detect all known variants and normalise to seconds.

    Returns:
      int   — resolved delay in seconds (always ≥ 1)
      str   — error message if no time argument found at all
    """
    # Seconds variants
    for key in ("delay_seconds", "delay_s", "seconds"):
        if key in args:
            return max(1, int(args[key]))
    # Minutes variants
    for key in ("delay_minutes", "delay_m", "minutes", "minutes_remaining"):
        if key in args:
            return max(1, int(args[key]) * 60)
    # Hours variants
    for key in ("delay_hours", "hours"):
        if key in args:
            return max(1, int(args[key]) * 3600)
    # None of the expected keys found
    return "Error: no time argument provided. Specify delay_seconds (e.g. delay_seconds: 30)."


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def _set_reminder(args: dict) -> str:
    global _next_id

    # Message is optional — default to "Reminder" so a missing arg never fails.
    message: str = (args.get("message") or "Reminder").strip()

    delay_result = _resolve_delay(args)
    if isinstance(delay_result, str):
        # _resolve_delay returned an error message
        return delay_result
    delay_s: int = delay_result

    # 24-hour safety guard — unusually far reminders are probably a unit mistake.
    if delay_s > 86_400:
        hours = delay_s / 3600
        return (
            f"That's {hours:.1f} hours away — did you mean seconds instead of minutes? "
            f"Reply with the correct time (e.g. 'remind me in 30 seconds to ...')."
        )

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
        "Set a reminder that fires after a delay. "
        "delay_seconds MUST be in seconds — convert any minutes or hours before calling. "
        "Examples: '30 seconds' → delay_seconds: 30, "
        "'2 minutes' → delay_seconds: 120, '1 hour' → delay_seconds: 3600. "
        "The reminder will be spoken aloud and shown as a notification."
    ),
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "What to remind the user about.",
            },
            "delay_seconds": {
                "type": "integer",
                "description": (
                    "How many seconds until the reminder fires. "
                    "Convert time units: 1 minute = 60 seconds, 1 hour = 3600 seconds. "
                    "Always use seconds. "
                    "Examples: '30 seconds' → 30, '2 minutes' → 120, '1 hour' → 3600."
                ),
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
