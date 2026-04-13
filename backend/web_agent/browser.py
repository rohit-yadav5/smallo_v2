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

    def __init__(self) -> None:
        self._playwright = None
        self._context: Optional[BrowserContext] = None  # IS the persistent browser
        self._page: Optional[Page] = None

    # ── Singleton access ──────────────────────────────────────────────────────

    @classmethod
    async def get(cls) -> "BrowserManager":
        """Return the singleton, launching the browser if not yet started."""
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
            headless=False,
            args=[
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-blink-features=AutomationControlled",  # reduces bot detection
            ],
            viewport={"width": 1280, "height": 720},
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
        if self._context is None:
            await self._launch()
            return self._page  # type: ignore[return-value]
        try:
            # Accessing .pages verifies the context is still alive.
            _ = self._context.pages
        except Exception:
            print("  [browser] context died — relaunching", flush=True)
            self._context = None
            self._page = None
            await self._launch()
        return self._page  # type: ignore[return-value]

    async def new_page(self) -> Page:
        """Open and return a secondary page (used by the monitor loop)."""
        if self._context is None:
            await self._launch()
        return await self._context.new_page()  # type: ignore[union-attr]

    async def screenshot_b64(self) -> str:
        """
        Capture the current viewport as JPEG and return as a base64 string.

        Uses JPEG at 75% quality to keep payloads small (~50–200 KB for a
        1280×720 viewport).  Falls back to a lower quality if the result
        exceeds 500 KB after encoding.
        """
        pg = await self.page()
        data = await pg.screenshot(type="jpeg", quality=75, full_page=False)
        if len(data) > 500_000:
            # Still too large — drop quality further
            data = await pg.screenshot(type="jpeg", quality=40, full_page=False)
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
