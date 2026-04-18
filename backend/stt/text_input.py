"""backend/stt/text_input.py — File-based text input watcher for Small O.

Polls backend/data/text_input.txt every 500 ms.  When the file contains
non-empty text it reads the content, wipes the file, then calls
``on_transcript(text)`` — the exact same callback used after Whisper STT
produces a result.  This means typed text goes through the full pipeline
(plugin routing → LLM → TTS) without touching the audio stack at all.

Usage
─────
This is designed as a background asyncio task started in backend/main.py:

    asyncio.create_task(watch_text_input(on_transcript=_enqueue_text))

The user drops text into the file however they like (shell redirect, editor,
another process) and the pipeline picks it up automatically.

Phase 2 note: this is also a useful debugging surface — you can inject
planner inputs or test utterances without needing a microphone.
"""

import asyncio
from pathlib import Path
from typing import Awaitable, Callable

WATCHED_FILE = Path(__file__).resolve().parent.parent / "data" / "text_input.txt"


def _ensure_file() -> None:
    """Create the watched file (and its parent directory) if it doesn't exist."""
    WATCHED_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not WATCHED_FILE.exists():
        WATCHED_FILE.write_text("", encoding="utf-8")


async def watch_text_input(
    on_transcript: Callable[[str], None],
    poll_interval_s: float = 0.5,
) -> None:
    """Continuously poll text_input.txt and forward non-empty content.

    Parameters
    ----------
    on_transcript:  Called with the stripped text when new input is found.
                    Must be a synchronous callable (not async) — it just
                    enqueues the text into the pipeline queue.
    poll_interval_s: How often to check the file (default 500 ms).
    """
    _ensure_file()
    print(
        f"  [text_input] watching {WATCHED_FILE}  "
        f"(poll every {poll_interval_s:.2f}s)",
        flush=True,
    )

    while True:
        try:
            content = WATCHED_FILE.read_text(encoding="utf-8").strip()
            if content:
                # Wipe the file immediately so we don't process the same
                # text twice if the loop fires before the user writes again.
                WATCHED_FILE.write_text("", encoding="utf-8")
                print(f"  [text_input] received: {content[:80]!r}", flush=True)
                on_transcript(content)
        except Exception as exc:
            # File I/O errors are transient (file being written) — skip silently.
            print(f"  [text_input] ⚠ poll error: {exc}", flush=True)

        await asyncio.sleep(poll_interval_s)
