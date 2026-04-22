"""backend/web_agent/browser.py — Persistent Playwright browser singleton.

Manages a single long-lived Chromium instance that:
  • Stays open across tasks (persistent BrowserContext, not ephemeral)
  • Stores its profile (cookies, localStorage, sessions) at PROFILE_DIR
    so the user stays logged into sites across server restarts
  • Runs headed (visible window) so the user can observe what the agent does
  • Is shared by all web_* tools — no parallel browser sessions

Usage
─────
    manager = await BrowserManager.get()
    page    = await manager.page()
    await page.goto("https://example.com")
    b64     = await manager.screenshot_b64()

The singleton is re-entrant: all tools call BrowserManager.get() which
returns the same instance; a new instance is created only if the browser
has been closed externally.
"""

import asyncio
import base64
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, BrowserContext, Page
from config.limits import BROWSER_WAIT_MS

# Profile directory — persists cookies, localStorage, and session tokens
# across server restarts.  Created at first launch if it does not exist.
PROFILE_DIR = Path(__file__).resolve().parent.parent / "data" / "browser_profile"


class BrowserManager:
    """
    Singleton managing one persistent Chromium BrowserContext.

    Uses launch_persistent_context() so the profile directory is bound to the
    Chromium process directly — no separate Browser + Context indirection needed.
    """

    _instance: Optional["BrowserManager"] = None
    _launch_lock: Optional[asyncio.Lock] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._launch_lock is None:
            cls._launch_lock = asyncio.Lock()
        return cls._launch_lock

    def __init__(self) -> None:
        self._playwright = None
        self._context: Optional[BrowserContext] = None  # IS the persistent browser
        self._page: Optional[Page] = None

    # ── Singleton access ──────────────────────────────────────────────────────

    @classmethod
    async def get(cls) -> "BrowserManager":
        """Return the singleton, launching the browser if not yet started."""
        async with cls._get_lock():
            if cls._instance is None:
                cls._instance = BrowserManager()
                await cls._instance._launch()
        return cls._instance

    # ── Internal launch ───────────────────────────────────────────────────────

    async def _launch(self) -> None:
        """Start Playwright and open the persistent Chromium context."""
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=True,
            args=[
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-blink-features=AutomationControlled",  # reduces bot detection
                "--disable-infobars",
                # ── Memory reduction flags ────────────────────────────────
                "--disable-extensions",
                "--disable-plugins",
                "--disable-background-networking",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-dev-shm-usage",
                "--js-flags=--max-old-space-size=256",    # cap JS heap to 256 MB
                "--memory-pressure-thresholds=critical=0.9",
            ],
            viewport={"width": 1280, "height": 800},
        )
        # Reuse an existing page if one is already open (e.g. after a reload),
        # otherwise open a fresh one.
        pages = self._context.pages
        self._page = pages[0] if pages else await self._context.new_page()
        print(
            f"  [browser] Chromium launched  profile={PROFILE_DIR}",
            flush=True,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def page(self) -> Page:
        """Return the active page, relaunching the browser if it was closed."""
        if self._context is not None:
            try:
                # Accessing .pages verifies the context is still alive.
                _ = self._context.pages
                return self._page  # type: ignore[return-value]
            except Exception:
                pass
        async with BrowserManager._get_lock():
            if self._context is None:
                print("  [browser] context died — relaunching", flush=True)
                await self._launch()
        return self._page  # type: ignore[return-value]

    async def new_page(self) -> Page:
        """Open and return a secondary page (used by the monitor loop)."""
        if self._context is None:
            async with BrowserManager._get_lock():
                if self._context is None:
                    await self._launch()
        return await self._context.new_page()  # type: ignore[union-attr]

    async def screenshot_b64(self) -> str:
        """
        Capture the current viewport and return as a base64 JPEG string.

        Always uses JPEG at quality=60 — ~70% smaller than PNG with no
        visible quality loss for the WebViewer use case.  Waits briefly
        for pending renders to settle before capturing.

        Strategy:
          1. Wait up to 2s for networkidle (catches late JS renders).
          2. Extra 300ms for final animation frames.
          3. Capture JPEG quality=60 clipped to 1280×800 viewport.
        """
        pg = await self.page()

        # Let any in-flight requests and render passes finish
        try:
            await pg.wait_for_load_state("networkidle", timeout=BROWSER_WAIT_MS)
        except Exception:
            pass   # live-updating pages never reach networkidle — that's fine
        await asyncio.sleep(0.3)

        data = await pg.screenshot(
            type="jpeg",
            quality=60,
            full_page=False,
            clip={"x": 0, "y": 0, "width": 1280, "height": 800},
        )
        return base64.b64encode(data).decode()

    async def shutdown(self) -> None:
        """Gracefully close the browser and the Playwright server process."""
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        BrowserManager._instance = None
        print("  [browser] Chromium shut down", flush=True)
