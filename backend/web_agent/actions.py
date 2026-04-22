"""backend/web_agent/actions.py — Raw async Playwright action primitives.

Each function takes a Playwright Page and keyword arguments, performs one
browser action, and returns a human-readable result string.  Errors are
surfaced as return strings (never raised) so the LLM can reason about them.

These are not registered as tools directly — agent.py wraps them with the
screenshot broadcast and registers them in the ToolRegistry.

Wait strategy
─────────────
navigate() uses a two-phase wait:
  1. domcontentloaded — fast baseline, always fires
  2. networkidle (5s cap) — catches most JS-rendered content
  3. 1.5s extra sleep — covers React hydration, Google AI overviews, etc.

search_google() adds a third phase:
  3. wait_for_selector("h3") — organic results must appear
  4. 3.0s sleep — AI overview loads after organic results
"""

import asyncio

from playwright.async_api import Page
from config.limits import TOOL_OUTPUT_BROWSER


async def navigate(page: Page, url: str) -> str:
    """
    Navigate to URL with smart wait strategy.

    1. domcontentloaded — fast, always fires
    2. networkidle (5s cap) — catches most JS rendering without hanging
       on pages with live feeds or websockets
    3. 1.5s extra sleep — covers React hydration, lazy hydration, etc.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=5_000)
        except Exception:
            pass   # live-updating pages never reach networkidle — that's ok
        await asyncio.sleep(1.5)
        title = await page.title()
        return f"Navigated to: {title} — {page.url}"
    except Exception as exc:
        return f"Navigation failed: {exc}"


async def click(page: Page, selector: str) -> str:
    """Click an element by CSS selector, then fall back to visible-text match."""
    try:
        await page.click(selector, timeout=5_000)
        return f"Clicked: {selector}"
    except Exception:
        try:
            await page.get_by_text(selector).first.click(timeout=5_000)
            return f"Clicked text: {selector}"
        except Exception as exc:
            return f"Click failed for {selector!r}: {exc}"


async def type_text(page: Page, selector: str, text: str) -> str:
    """Click a form field and type text into it."""
    try:
        await page.click(selector, timeout=5_000)
        await page.fill(selector, text)
        return f"Typed into {selector}: {text[:40]}"
    except Exception as exc:
        return f"Type failed for {selector!r}: {exc}"


async def get_page_text(page: Page, max_chars: int = TOOL_OUTPUT_BROWSER) -> str:
    """
    Extract visible text with noise reduction and lazy-load triggering.

    Steps:
      1. Scroll to bottom to trigger lazy-loaded images/content.
      2. Wait 800ms for any triggered renders to settle.
      3. Scroll back to top.
      4. Extract innerText while temporarily marking nav/footer/cookie
         elements so they don't pollute the content.

    Uses innerText (visibility-aware) not textContent (includes hidden).
    """
    try:
        # Trigger lazy-load by scrolling to bottom then back
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(0.8)
        await page.evaluate("window.scrollTo(0, 0)")

        text: str = await page.evaluate("""
            () => {
                // Temporarily tag noisy structural elements
                const noiseSelectors = [
                    'nav', 'footer', 'header',
                    '[role="banner"]', '[role="navigation"]',
                    '.cookie-banner', '#cookie-notice',
                    '[aria-hidden="true"]'
                ];
                const tagged = [];
                noiseSelectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => {
                        el.setAttribute('data-smallo-hidden', 'true');
                        tagged.push(el);
                    });
                });

                const text = document.body.innerText;

                // Restore — remove the temporary attribute
                tagged.forEach(el => el.removeAttribute('data-smallo-hidden'));

                return text;
            }
        """)

        if len(text) > max_chars:
            return text[:max_chars] + f"\n[...truncated, {len(text)} chars total]"
        return text
    except Exception as exc:
        return f"get_page_text failed: {exc}"


async def get_page_links(page: Page) -> str:
    """Return up to 100 links on the current page as 'url | link text' lines."""
    try:
        links: list[str] = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]'))
                       .map(a => a.href + ' | ' + (a.innerText || a.title || '').trim())
                       .filter(l => l.length > 4)
        """)
        return "\n".join(links[:100])
    except Exception as exc:
        return f"get_page_links failed: {exc}"


async def scroll(page: Page, direction: str = "down", amount: int = 3) -> str:
    """Scroll the page up or down by `amount` screen-heights."""
    pixels = amount * 600
    delta  = pixels if direction == "down" else -pixels
    try:
        await page.evaluate(f"window.scrollBy(0, {delta})")
        return f"Scrolled {direction} {amount} screen(s)"
    except Exception as exc:
        return f"Scroll failed: {exc}"


async def wait_for(page: Page, selector: str, timeout_ms: int = 10_000) -> str:
    """Wait for a CSS selector to appear on the page."""
    try:
        await page.wait_for_selector(selector, timeout=timeout_ms)
        return f"Element appeared: {selector}"
    except Exception as exc:
        return f"wait_for failed for {selector!r}: {exc}"


async def press_key(page: Page, key: str) -> str:
    """Press a keyboard key (e.g. 'Enter', 'Tab', 'Escape')."""
    try:
        await page.keyboard.press(key)
        return f"Pressed: {key}"
    except Exception as exc:
        return f"press_key failed for {key!r}: {exc}"


async def get_current_url(page: Page) -> str:
    """Return the current page URL."""
    return page.url


async def search_google(page: Page, query: str) -> str:
    """
    Search Google and wait for AI overview + organic results.

    Wait phases:
      1. domcontentloaded — page skeleton ready
      2. wait_for_selector("h3", 5s) — organic result titles must appear
      3. 3.0s sleep — Google AI overview loads after organic results

    Attempts to extract AI overview text from known Google selectors.
    Falls back gracefully if the overview is absent or the selector changes.
    """
    try:
        encoded = query.replace(" ", "+")
        await page.goto(
            f"https://www.google.com/search?q={encoded}",
            wait_until="domcontentloaded",
            timeout=30_000,
        )

        # Wait for organic results to appear
        try:
            await page.wait_for_selector("h3", timeout=5_000)
        except Exception:
            pass

        # Extra wait for AI overview (loads asynchronously after organic results)
        await asyncio.sleep(3.0)

        # Try to extract AI overview text from known Google selectors
        ai_overview: str | None = await page.evaluate("""
            () => {
                const selectors = [
                    '[data-attrid="wa:/description"]',
                    '.KBlKJd',
                    '[jsname="yEVEwb"]',
                    '.IVvPP',
                    'div[class*="AIs"] p',
                ];
                for (const sel of selectors) {
                    const el = document.querySelector(sel);
                    if (el && el.innerText && el.innerText.length > 50) {
                        return 'AI Overview: ' + el.innerText.slice(0, 1000);
                    }
                }
                return null;
            }
        """)

        # Extract organic result titles
        titles: list[str] = await page.evaluate("""
            () => Array.from(document.querySelectorAll('h3'))
                       .map(h => h.innerText)
                       .filter(t => t.length > 5)
                       .slice(0, 10)
        """)

        # Extract result URLs
        links: list[str] = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href^="http"]'))
                       .map(a => a.href)
                       .filter(h => !h.includes('google.com'))
                       .slice(0, 10)
        """)

        combined = [f"{t} — {l}" for t, l in zip(titles, links)]
        output = "\n".join(combined) if combined else "No results found."

        if ai_overview:
            output = ai_overview + "\n\n" + output

        return output

    except Exception as exc:
        return f"Google search failed: {exc}"
