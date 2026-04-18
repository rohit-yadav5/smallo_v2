"""backend/bot_docs/store.py — Manages the bot-docs/ directory.

Small O's personal file store: all bot-generated files land here with
UID filenames. Human-readable titles are stored in a JSON index alongside.
Files are tagged by session_id so the frontend can filter to the current run.
"""

import json
import os
import pathlib
import random
import shutil
import string
import time
from dataclasses import asdict, dataclass
from typing import Optional

BOT_DOCS_DIR = pathlib.Path(__file__).parent.parent.parent / "bot-docs"
INDEX_FILE   = BOT_DOCS_DIR / "_index.json"
UID_CHARS    = string.ascii_lowercase
UID_LENGTH   = 20


@dataclass
class DocEntry:
    uid:        str    # e.g. "abcdefghijklmnopqrst"
    filename:   str    # e.g. "abcdefghijklmnopqrst.txt"
    title:      str    # human readable, e.g. "Python research notes"
    extension:  str    # ".txt", ".md", ".py", etc.
    created_at: float  # unix timestamp
    size_bytes: int
    session_id: str    # ties file to a session for frontend filtering


def _generate_uid() -> str:
    """Generate a 20-char lowercase alphabetic UID. Collision-safe."""
    while True:
        uid = "".join(random.choices(UID_CHARS, k=UID_LENGTH))
        # Check both possible extensions to avoid any collision
        if not list(BOT_DOCS_DIR.glob(f"{uid}.*")):
            return uid


def _load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        return []
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_index(entries: list[dict]) -> None:
    INDEX_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def ensure_dirs() -> None:
    """Create bot-docs/ directory if missing. Call at startup."""
    BOT_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def save_file(
    content: str,
    title: str,
    extension: str = ".txt",
    session_id: str = "default",
    also_copy_to: Optional[str] = None,
) -> DocEntry:
    """Save content to bot-docs/ with a UID filename.

    If also_copy_to is set (e.g. ~/Desktop/notes.txt), also writes a copy
    there — but bot-docs/ is always the primary store.
    Returns the DocEntry for this file.
    """
    ensure_dirs()
    uid = _generate_uid()
    ext = extension if extension.startswith(".") else f".{extension}"
    filename = f"{uid}{ext}"
    filepath = BOT_DOCS_DIR / filename

    filepath.write_text(content, encoding="utf-8")
    size = filepath.stat().st_size

    entry = DocEntry(
        uid=uid,
        filename=filename,
        title=title,
        extension=ext,
        created_at=time.time(),
        size_bytes=size,
        session_id=session_id,
    )

    index = _load_index()
    index.append(asdict(entry))
    _save_index(index)

    if also_copy_to:
        dest = pathlib.Path(os.path.expanduser(also_copy_to))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filepath, dest)

    return entry


def get_session_files(session_id: str) -> list[DocEntry]:
    """Return all files created in this session, newest first."""
    index = _load_index()
    entries = [DocEntry(**e) for e in index if e["session_id"] == session_id]
    return sorted(entries, key=lambda e: e.created_at, reverse=True)


def get_file_path(uid: str) -> Optional[pathlib.Path]:
    """Return the absolute path to a file by UID, or None if not found."""
    index = _load_index()
    for e in index:
        if e["uid"] == uid:
            ext = e["extension"]
            p = BOT_DOCS_DIR / f"{uid}{ext}"
            return p if p.exists() else None
    return None


def get_file_content(uid: str) -> Optional[str]:
    """Return file content by UID, or None if not found."""
    path = get_file_path(uid)
    if path:
        return path.read_text(encoding="utf-8")
    return None


def get_entry_by_uid(uid: str) -> Optional[DocEntry]:
    """Return the DocEntry for a given UID, or None if not found."""
    index = _load_index()
    for e in index:
        if e["uid"] == uid:
            return DocEntry(**e)
    return None
