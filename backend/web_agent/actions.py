"""backend/web_agent/actions.py — Raw async Playwright action primitives.

Each function takes a Playwright Page and keyword arguments, performs one
browser action, and returns a human-readable result string.  Errors are
surfaced as return strings (never raised) so the LLM can reason about them.

These are not registered as tools directly — agent.py wraps them with the
screenshot broadcast and registers them in the ToolRegistry.
"""

from playwright.async_api import Page


async def navigate(page: Page, url: str) -> str:
    """Navigate to a URL.  Wait for the DOM to load.  Returns title + final URL."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        return f"Navigated to: {page.title()} — {page.url}"
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


async def get_page_text(page: Page, max_chars: int = 8_000) -> str:
    """Extract visible body text via JS innerText.  Caps at max_chars."""
    try:
        text: str = await page.evaluate("() => document.body.innerText")
        return text[:max_chars] if len(text) > max_chars else text
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
    Navigate to Google, run a search, and return the top result titles + URLs.
    Returns up to 10 results as 'Title — URL' lines.
    """
    try:
        encoded = query.replace(" ", "+")
        await page.goto(
            f"https://www.google.com/search?q={encoded}",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        titles: list[str] = await page.evaluate("""
            () => Array.from(document.querySelectorAll('h3'))
                       .map(h => h.innerText)
                       .filter(t => t.length > 5)
                       .slice(0, 10)
        """)
        links: list[str] = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href^="http"]'))
                       .map(a => a.href)
                       .filter(h => !h.includes('google.com'))
                       .slice(0, 10)
        """)
        combined = [
            f"{t} — {l}" for t, l in zip(titles, links)
        ]
        return "\n".join(combined) if combined else "No results found."
    except Exception as exc:
        return f"Google search failed: {exc}"
