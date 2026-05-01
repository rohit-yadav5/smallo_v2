"""backend/mode.py — Conversation mode state.

Two modes:
  * "normal"  — stateless chat. No memory writes. Memory retrieval skipped.
                Continuity within a session via session_history deque.
  * "super"   — full memory system active (legacy behaviour).

State is in-memory only and resets to "normal" on each backend boot.
The frontend persists user preference in localStorage and re-sends it
on connect.
"""

from typing import Callable

_MODE: str = "normal"
_HANDLERS: list[Callable[[str, str], None]] = []

_VALID_MODES = ("normal", "super")


def get_mode() -> str:
    return _MODE


def is_super() -> bool:
    return _MODE == "super"


def is_normal() -> bool:
    return _MODE == "normal"


def set_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; expected one of {_VALID_MODES}")
    global _MODE
    old = _MODE
    if old == mode:
        return
    _MODE = mode
    for fn in list(_HANDLERS):
        try:
            fn(old, mode)
        except Exception as exc:
            print(f"  [mode] change handler failed (non-fatal): {exc}", flush=True)


def register_mode_change_handler(fn: Callable[[str, str], None]) -> None:
    _HANDLERS.append(fn)
