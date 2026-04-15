"""backend/tools/file_tool.py — File read/write tools for Small O.

Registers two tools at import time:
  • read_file  — read any file the user can access (capped at 50 k chars)
  • write_file — create or append to any file (creates parent dirs automatically)

Security note: these tools run with the same OS permissions as the Python
process.  No additional sandboxing is applied in Phase 1.  Phase 2 may add a
path-allowlist check before executing.
"""

import asyncio
import os
import pathlib
import re
from pathlib import Path

from tools.registry import ToolDefinition, registry


_MAX_CHARS = 50_000

# Compiled once at import time — matches common LLM-hallucinated placeholders.
_PLACEHOLDER_RE = re.compile(
    r"(?:/Users/[Yy]our[Uu]sername"
    r"|/Users/\[username\]"
    r"|/home/[Yy]our[Uu]sername"
    r"|\$HOME"
    r"|\$\{HOME\})",
)


def resolve_path(path: str) -> str:
    """Replace LLM-generated username placeholders with the real home directory.

    Pure function — no side effects, no I/O.  Handles patterns such as:
      /Users/YourUsername/…  →  /Users/rohit/…
      /Users/yourUsername/…  →  /Users/rohit/…
      /Users/[username]/…    →  /Users/rohit/…
      $HOME/…                →  /Users/rohit/…
      ${HOME}/…              →  /Users/rohit/…
      ~/…                    →  expanded by os.path.expanduser

    The result is then passed through expanduser() to handle any bare ~ prefixes.
    """
    home = str(pathlib.Path.home())
    resolved = _PLACEHOLDER_RE.sub(home, path)
    return os.path.expanduser(resolved)


async def _read_file(args: dict) -> str:
    path_str: str = args.get("path", "").strip()
    if not path_str:
        return "Error: 'path' argument is required."

    path = Path(resolve_path(path_str)).resolve()

    try:
        if not path.exists():
            return f"Error: file not found at {path}"
        if path.is_dir():
            return f"Error: {path} is a directory, not a file."

        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > _MAX_CHARS:
            truncated = content[:_MAX_CHARS]
            notice = (
                f"\n\n[Truncated: file is {len(content):,} chars; "
                f"showing first {_MAX_CHARS:,} chars only.]"
            )
            return truncated + notice
        return content
    except PermissionError:
        return f"Error: permission denied reading {path}"
    except Exception as exc:
        return f"Error reading {path}: {exc}"


async def _write_file(args: dict) -> str:
    path_str: str = args.get("path", "").strip()
    content: str  = args.get("content", "")
    mode: str     = args.get("mode", "overwrite").lower()

    if not path_str:
        return "Error: 'path' argument is required."
    if mode not in ("overwrite", "append"):
        mode = "overwrite"

    path = Path(resolve_path(path_str)).resolve()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if mode == "append" else "w"
        with path.open(file_mode, encoding="utf-8") as fh:
            fh.write(content)
        return f"Written {len(content):,} chars to {path} (mode={mode})"
    except PermissionError:
        return f"Error: permission denied writing to {path}"
    except Exception as exc:
        return f"Error writing to {path}: {exc}"


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name="read_file",
    description="Read the contents of a file at the given path. Returns up to 50,000 characters.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or ~-relative path to the file to read.",
            },
        },
        "required": ["path"],
    },
    handler=_read_file,
))

registry.register(ToolDefinition(
    name="write_file",
    description="Write or append text content to a file. Creates parent directories automatically.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or ~-relative path to write to.",
            },
            "content": {
                "type": "string",
                "description": "The text content to write.",
            },
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
                "description": "Whether to overwrite the file (default) or append to it.",
            },
        },
        "required": ["path", "content"],
    },
    handler=_write_file,
))
