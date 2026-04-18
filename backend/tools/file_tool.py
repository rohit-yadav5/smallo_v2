"""backend/tools/file_tool.py — File read/write tools for Small O.

Registers two tools at import time:
  • read_file  — read any file the user can access (capped at 50 k chars)
  • write_file — save content to the bot-docs/ managed store (UID filenames,
                 human-readable titles, JSON index).  Optionally also copies
                 to a user-specified path.

Security note: these tools run with the same OS permissions as the Python
process.  No additional sandboxing is applied in Phase 1.
"""

import os
import pathlib
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

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

# Set by main.py at startup via set_broadcast_fn() / set_session_id().
_broadcast_fn: Optional[Callable] = None
_session_id: str = "default"


def set_broadcast_fn(fn: Callable) -> None:
    """Wire in the _emit broadcast function from main.py."""
    global _broadcast_fn
    _broadcast_fn = fn


def set_session_id(sid: str) -> None:
    """Set the session ID for all files saved this run."""
    global _session_id
    _session_id = sid


def resolve_path(path: str) -> str:
    """Replace LLM-generated username placeholders with the real home directory.

    Pure function — no side effects, no I/O.  Handles patterns such as:
      /Users/YourUsername/…  →  /Users/rohit/…
      /Users/yourUsername/…  →  /Users/rohit/…
      /Users/[username]/…    →  /Users/rohit/…
      $HOME/…                →  /Users/rohit/…
      ${HOME}/…              →  /Users/rohit/…
      ~/…                    →  expanded by os.path.expanduser
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
    content:   str = args.get("content", "")
    title:     str = args.get("title", "").strip()
    path_str:  str = args.get("path", "").strip()
    ext:       str = args.get("extension", ".txt")

    # Infer extension from explicit path if given
    if path_str:
        _, inferred_ext = os.path.splitext(path_str)
        if inferred_ext:
            ext = inferred_ext

    # Normalise extension
    if not ext.startswith("."):
        ext = f".{ext}"

    # Auto-generate title from first non-empty content line if not provided
    if not title:
        first_line = next(
            (l.strip() for l in content.splitlines() if l.strip()), ""
        )
        title = first_line[:60] or "Untitled document"

    # Determine whether to also copy to an external path
    also_copy_to: Optional[str] = None
    if path_str:
        resolved = resolve_path(path_str)
        # Only copy externally when path is NOT inside bot-docs
        if "bot-docs" not in resolved:
            also_copy_to = resolved

    from bot_docs.store import save_file

    entry = save_file(
        content=content,
        title=title,
        extension=ext,
        session_id=_session_id,
        also_copy_to=also_copy_to,
    )

    # Broadcast FILE_CREATED event to all connected frontend clients
    if _broadcast_fn is not None:
        _broadcast_fn("FILE_CREATED", asdict(entry))

    result = f"Saved '{entry.title}' → bot-docs/{entry.filename}"
    if also_copy_to:
        result += f" (also copied to {also_copy_to})"
    return result


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
    description=(
        "Create a file with content. "
        "Provide a human-readable 'title' describing the content "
        "(e.g. 'Python research notes', 'App ideas brainstorm'). "
        "Optionally provide 'path' if the user explicitly requested a specific "
        "save location — a copy will also be kept in bot-docs/. "
        "Optionally provide 'extension' (e.g. '.md', '.py') to set the file type."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The text content to write.",
            },
            "title": {
                "type": "string",
                "description": (
                    "Human-readable title for this file "
                    "(e.g. 'Ocean poem', 'Shopping list'). "
                    "Never include the UID or filename — titles are for humans."
                ),
            },
            "extension": {
                "type": "string",
                "description": "File extension including dot, e.g. '.txt', '.md', '.py'. Defaults to '.txt'.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Optional explicit save path (absolute or ~-relative). "
                    "Use only when the user specifically requested a location. "
                    "A copy is always kept in bot-docs/ regardless."
                ),
            },
        },
        "required": ["content"],
    },
    handler=_write_file,
))
