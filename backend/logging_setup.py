"""backend/logging_setup.py — stdlib logging with colored console + rotating file.

Env vars:
  SMALLO_LOG_LEVEL  one of DEBUG/INFO/WARNING/ERROR  (default INFO)
  SMALLO_LOG_FILE   "0" to disable file logging      (default "1")

Log lines follow:  HH:MM:SS.mmm LEVEL [name] event_name k1=v1 k2=v2
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR  = Path(__file__).resolve().parent / "logs"
_LOG_FILE = _LOG_DIR / "smallo.log"

_COLORS = {
    "DEBUG":    "\033[2;37m",   # dim grey
    "INFO":     "\033[36m",     # cyan
    "WARNING":  "\033[33m",     # yellow
    "ERROR":    "\033[31m",     # red
    "CRITICAL": "\033[1;31m",   # bold red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool) -> None:
        super().__init__("%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s",
                         datefmt="%H:%M:%S")
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        if not self._use_color:
            return line
        color = _COLORS.get(record.levelname, "")
        return f"{color}{line}{_RESET}" if color else line


_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    level_name = os.environ.get("SMALLO_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ColorFormatter(use_color=sys.stdout.isatty()))
    root.addHandler(console)

    if os.environ.get("SMALLO_LOG_FILE", "1") != "0":
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(_ColorFormatter(use_color=False))
        root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
