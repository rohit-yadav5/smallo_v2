"""backend/session_history.py — In-memory conversation deque (normal mode).

Holds the last N turns of user/assistant exchanges to preserve continuity
within a session without writing to long-term memory. Cleared on mode
change. Not used in super mode (memory system handles continuity there).
"""

import time
from collections import deque

_MAXLEN = 20
_HISTORY: deque[dict] = deque(maxlen=_MAXLEN)


def append(role: str, content: str) -> None:
    if role not in ("user", "assistant"):
        raise ValueError(f"invalid role {role!r}")
    if not content:
        return
    _HISTORY.append({"role": role, "content": content, "timestamp": time.time()})


def get_messages() -> list[dict]:
    """Return a list copy shaped for direct LLM consumption."""
    return [{"role": m["role"], "content": m["content"]} for m in _HISTORY]


def clear() -> None:
    _HISTORY.clear()


def count() -> int:
    return len(_HISTORY)
