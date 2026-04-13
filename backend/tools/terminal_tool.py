"""backend/tools/terminal_tool.py — Shell command execution tool for Small O.

Registers one tool at import time:
  • run_terminal — run an arbitrary shell command and return its output.

Safety
──────
• A blocklist of known destructive patterns is checked before any execution.
• Commands are killed hard after timeout_s seconds (default 30).
• Every invocation is logged with timestamp + command to
  backend/logs/terminal_history.log for auditing.

Phase 2 note: this tool is the primitive for a future "DevOps agent" that can
run build scripts, tests, and deployments.  The blocklist should be extended
before giving that agent access to production machines.
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path

from tools.registry import ToolDefinition, registry


# ── Logging ──────────────────────────────────────────────────────────────────

_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "terminal_history.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _log_command(command: str, exit_code: int | None) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    status = f"exit={exit_code}" if exit_code is not None else "killed"
    try:
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] [{status}] {command}\n")
    except Exception:
        pass  # never let logging break the tool


# ── Blocklist ─────────────────────────────────────────────────────────────────

_BLOCKLIST: list[str] = [
    r"rm\s+-[rRf]*\s+/",       # rm -rf /
    r"rm\s+-[rRf]*\s+~",       # rm -rf ~
    r"sudo\s+rm",              # sudo rm anything
    r"mkfs",                   # format filesystem
    r"dd\s+if=",               # raw disk write
    r":\(\)\s*\{.*\}",         # fork bomb :(){:|:&};:
    r">\s*/dev/sd",            # overwrite block device
    r"chmod\s+-R\s+777\s+/",   # chmod 777 root
]

_BLOCKLIST_RE = [re.compile(p, re.IGNORECASE) for p in _BLOCKLIST]


def _is_blocked(command: str) -> bool:
    return any(rx.search(command) for rx in _BLOCKLIST_RE)


# ── Handler ──────────────────────────────────────────────────────────────────

async def _run_terminal(args: dict) -> str:
    command: str   = args.get("command", "").strip()
    timeout_s: int = int(args.get("timeout_s", 30))

    if not command:
        return "Error: 'command' argument is required."

    if _is_blocked(command):
        _log_command(command, None)
        return "Blocked: unsafe command pattern detected."

    # Clamp timeout to a sane range
    timeout_s = max(1, min(timeout_s, 120))

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=float(timeout_s)
        )
        exit_code = proc.returncode
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        _log_command(command, None)
        return f"Error: command timed out after {timeout_s}s and was killed."

    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    stderr = stderr_b.decode("utf-8", errors="replace").strip()

    _log_command(command, exit_code)

    combined = stdout or stderr or "(no output)"
    # Cap output to avoid bloating the LLM context
    if len(combined) > 6_000:
        combined = combined[:6_000] + "\n[Output truncated]"

    return f"Exit {exit_code}:\n{combined}"


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name="run_terminal",
    description=(
        "Execute a shell command and return its output. "
        "Use for running scripts, checking system state, git operations, etc. "
        "Destructive commands (rm -rf, mkfs, etc.) are blocked."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run.",
            },
            "timeout_s": {
                "type": "integer",
                "description": "Maximum seconds to wait before killing the process. Default 30, max 120.",
            },
        },
        "required": ["command"],
    },
    handler=_run_terminal,
))
