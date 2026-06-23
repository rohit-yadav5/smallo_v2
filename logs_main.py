"""logs_main.py – Same as main.py, but with SMALLO_LOG_LEVEL=DEBUG and
the backend's rotating log file tailed live to stdout.

Usage:
    python logs_main.py
"""
import os

# Must be set before main.py spawns the backend subprocess so the env is
# inherited via os.environ.copy() in main.main().
os.environ["SMALLO_LOG_LEVEL"] = "DEBUG"
os.environ["SMALLO_LOG_FILE"]  = "1"

import threading
import time
from pathlib import Path

import main as _main_module

LOG_FILE = Path(__file__).resolve().parent / "backend" / "logs" / "smallo.log"


def _tail_log() -> None:
    """Poll for LOG_FILE to appear, then stream appended lines to stdout."""
    deadline = time.time() + 5.0
    while not LOG_FILE.exists():
        if time.time() > deadline:
            print(f"  [log] WARNING: {LOG_FILE} did not appear within 5s — not tailing")
            return
        time.sleep(0.1)

    try:
        with LOG_FILE.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)   # jump to EOF; only show NEW output
            while True:
                chunk = f.read()
                if chunk:
                    for line in chunk.splitlines():
                        print(f"[log] {line}", flush=True)
                else:
                    time.sleep(0.2)
    except Exception as exc:
        print(f"  [log] tail thread crashed: {exc}", flush=True)


if __name__ == "__main__":
    threading.Thread(target=_tail_log, daemon=True, name="log-tail").start()
    _main_module.main()
