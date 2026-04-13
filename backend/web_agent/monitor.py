"""backend/web_agent/monitor.py — Background webpage-change detector.

Periodically polls registered URLs, hashes their visible text, and fires a
PROACTIVE_EVENT when content changes.  Targets persist to
backend/data/monitor_targets.json so they survive server restarts.

Architecture
────────────
  • WebMonitor is a singleton instantiated as `web_monitor` at module level.
  • Call `web_monitor.run_forever(loop)` once from main.py after startup; it
    schedules an asyncio task that loops indefinitely.
  • Uses httpx (async HTTP) — NOT the shared Playwright browser — so checks
    never interrupt the user's browsing session.
  • Fires PROACTIVE_EVENT via the broadcast function registered with
    `set_broadcast_fn()` (wired from main.py the same way as agent.py).

PROACTIVE_EVENT payload
────────────────────────
  {
    "event":       "web_monitor",
    "target_id":   "uuid",
    "description": "human label set at registration",
    "summary":     "first 500 chars of changed content",
    "url":         "https://...",
  }

Tool registration
─────────────────
  monitor_add    — add a URL to watch
  monitor_remove — remove a URL by id
  monitor_list   — list all active targets
"""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx

from tools.registry import registry, ToolDefinition

# ── Persistence path ──────────────────────────────────────────────────────────

_DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
_STORE_PATH = _DATA_DIR / "monitor_targets.json"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class MonitorTarget:
    id:               str
    url:              str
    description:      str
    keywords:         list[str]        = field(default_factory=list)
    check_interval_s: int              = 60
    last_seen_hash:   Optional[str]    = None
    last_checked:     Optional[float]  = None   # epoch seconds


# ── WebMonitor singleton ──────────────────────────────────────────────────────

class WebMonitor:
    """Polls registered targets and fires PROACTIVE_EVENTs on change."""

    def __init__(self) -> None:
        self._targets: list[MonitorTarget] = []
        self._broadcast_fn: Optional[Callable] = None
        self._task: Optional[asyncio.Task] = None
        self._load()

    # ── Broadcast hook ────────────────────────────────────────────────────────

    def set_broadcast_fn(self, fn: Callable) -> None:
        self._broadcast_fn = fn

    def _fire(self, payload: dict) -> None:
        if self._broadcast_fn is not None:
            try:
                self._broadcast_fn("PROACTIVE_EVENT", payload)
            except Exception as exc:
                print(f"  [monitor] broadcast failed: {exc}", flush=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not _STORE_PATH.exists():
            return
        try:
            raw = json.loads(_STORE_PATH.read_text())
            self._targets = [MonitorTarget(**t) for t in raw]
            print(f"  [monitor] loaded {len(self._targets)} target(s)", flush=True)
        except Exception as exc:
            print(f"  [monitor] load failed ({exc}) — starting fresh", flush=True)
            self._targets = []

    def _save(self) -> None:
        try:
            _STORE_PATH.write_text(json.dumps([asdict(t) for t in self._targets], indent=2))
        except Exception as exc:
            print(f"  [monitor] save failed: {exc}", flush=True)

    # ── Target management ─────────────────────────────────────────────────────

    def add_target(
        self,
        url: str,
        description: str,
        keywords: list[str] | None = None,
        check_interval_s: int = 60,
    ) -> MonitorTarget:
        target = MonitorTarget(
            id               = str(uuid.uuid4()),
            url              = url,
            description      = description,
            keywords         = keywords or [],
            check_interval_s = check_interval_s,
        )
        self._targets.append(target)
        self._save()
        print(f"  [monitor] added target {target.id} — {url}", flush=True)
        return target

    def remove_target(self, target_id: str) -> bool:
        before = len(self._targets)
        self._targets = [t for t in self._targets if t.id != target_id]
        if len(self._targets) < before:
            self._save()
            print(f"  [monitor] removed target {target_id}", flush=True)
            return True
        return False

    def list_targets(self) -> list[MonitorTarget]:
        return list(self._targets)

    # ── Content fetching ──────────────────────────────────────────────────────

    async def _fetch_text(self, url: str) -> Optional[str]:
        """Fetch a URL and return stripped text content (no Playwright)."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        try:
            async with httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                headers=headers,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                # Very light HTML→text: strip tags via simple regex-free approach
                text = resp.text
                # Remove script/style blocks
                import re
                text = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", text, flags=re.DOTALL | re.IGNORECASE)
                # Strip remaining tags
                text = re.sub(r"<[^>]+>", " ", text)
                # Collapse whitespace
                text = re.sub(r"\s+", " ", text).strip()
                return text
        except Exception as exc:
            print(f"  [monitor] fetch failed for {url}: {exc}", flush=True)
            return None

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    # ── Background loop ───────────────────────────────────────────────────────

    async def _check_target(self, target: MonitorTarget) -> None:
        """Check one target; fire event if content changed."""
        text = await self._fetch_text(target.url)
        if text is None:
            return

        # Keyword filter: if keywords set, only watch if at least one present
        if target.keywords:
            lower = text.lower()
            if not any(kw.lower() in lower for kw in target.keywords):
                target.last_checked = time.time()
                return

        new_hash = self._hash(text)
        target.last_checked = time.time()

        if target.last_seen_hash is None:
            # First check — baseline; don't fire
            target.last_seen_hash = new_hash
            self._save()
            print(f"  [monitor] baseline set for {target.id} ({target.url})", flush=True)
            return

        if new_hash != target.last_seen_hash:
            target.last_seen_hash = new_hash
            self._save()
            summary = text[:500]
            print(f"  [monitor] change detected: {target.id} — {target.url}", flush=True)
            self._fire({
                "event":       "web_monitor",
                "target_id":   target.id,
                "description": target.description,
                "summary":     summary,
                "url":         target.url,
            })

    async def _loop(self) -> None:
        """Main polling loop — runs indefinitely."""
        print("  [monitor] background loop started", flush=True)
        while True:
            now = time.time()
            for target in list(self._targets):
                due = (
                    target.last_checked is None
                    or (now - target.last_checked) >= target.check_interval_s
                )
                if due:
                    try:
                        await self._check_target(target)
                    except Exception as exc:
                        print(f"  [monitor] error checking {target.id}: {exc}", flush=True)
            await asyncio.sleep(15)   # poll every 15s; per-target interval controls actual checks

    def run_forever(self, loop: asyncio.AbstractEventLoop) -> None:
        """Schedule the monitor loop as an asyncio task on `loop`."""
        self._task = loop.create_task(self._loop())


# Module-level singleton — imported by __init__.py
web_monitor = WebMonitor()


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def _monitor_add(args: dict) -> str:
    url              = args.get("url", "")
    description      = args.get("description", url)
    keywords         = args.get("keywords", [])
    check_interval_s = int(args.get("check_interval_s", 60))
    if not url:
        return "Error: url is required."
    target = web_monitor.add_target(url, description, keywords, check_interval_s)
    return (
        f"Now monitoring: {url}\n"
        f"  id:          {target.id}\n"
        f"  description: {description}\n"
        f"  interval:    {check_interval_s}s\n"
        f"  keywords:    {keywords or '(any change)'}"
    )


async def _monitor_remove(args: dict) -> str:
    target_id = args.get("id", "")
    if not target_id:
        return "Error: id is required."
    removed = web_monitor.remove_target(target_id)
    return f"Removed monitor {target_id}." if removed else f"No target with id {target_id!r}."


async def _monitor_list(args: dict) -> str:
    targets = web_monitor.list_targets()
    if not targets:
        return "No active monitors."
    lines = []
    for t in targets:
        last = (
            f"{int(time.time() - t.last_checked)}s ago"
            if t.last_checked else "never"
        )
        lines.append(
            f"• {t.id[:8]}…  {t.url}\n"
            f"  desc: {t.description} | interval: {t.check_interval_s}s | "
            f"keywords: {t.keywords or 'any'} | last checked: {last}"
        )
    return "\n".join(lines)


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name        = "monitor_add",
    description = (
        "Start monitoring a webpage for changes.  "
        "Fires a proactive alert when the page content changes."
    ),
    parameters  = {
        "type": "object",
        "properties": {
            "url":              {"type": "string",  "description": "URL to monitor"},
            "description":      {"type": "string",  "description": "Human-readable label"},
            "keywords":         {
                "type":        "array",
                "items":       {"type": "string"},
                "description": "Only alert if these keywords are present (optional — omit to alert on any change)",
            },
            "check_interval_s": {
                "type":        "integer",
                "description": "Check interval in seconds (default 60)",
                "default":     60,
            },
        },
        "required": ["url"],
    },
    handler = _monitor_add,
))

registry.register(ToolDefinition(
    name        = "monitor_remove",
    description = "Stop monitoring a webpage by its monitor ID",
    parameters  = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Monitor ID returned by monitor_add"},
        },
        "required": ["id"],
    },
    handler = _monitor_remove,
))

registry.register(ToolDefinition(
    name        = "monitor_list",
    description = "List all active webpage monitors",
    parameters  = {"type": "object", "properties": {}},
    handler     = _monitor_list,
))
