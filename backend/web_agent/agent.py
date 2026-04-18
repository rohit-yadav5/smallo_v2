"""backend/web_agent/agent.py — ToolRegistry integration for the web agent.

Wraps every raw Playwright action from actions.py as an async ToolRegistry
handler, then self-registers at import time so importing this module is
sufficient to make all web_* tools available to both the LLM turn path and
the planner.

Screenshot flow
───────────────
After every browser-mutating action (navigate, click, type, scroll, …) a
viewport screenshot is taken and broadcast to all connected WS clients as a
WEB_SCREENSHOT event.  The frontend WebViewer component displays it live.
Non-mutating actions (web_read, web_links, web_wait) skip the screenshot to
avoid flooding the channel.

Broadcast setup
───────────────
main.py calls set_broadcast_fn(_emit) after startup so screenshots reach the
frontend.  Before that call screenshots are silently skipped (no crash).
"""

import time
from typing import Callable, Optional

from web_agent.browser import BrowserManager
from web_agent.actions import (
    navigate, click, type_text, get_page_text,
    get_page_links, scroll, wait_for, press_key,
    get_current_url, search_google,
)
from tools.registry import registry, ToolDefinition


# ── Broadcast hook ────────────────────────────────────────────────────────────

_broadcast_fn: Optional[Callable] = None


def set_broadcast_fn(fn: Callable) -> None:
    """Register the _emit callback from main.py for WEB_SCREENSHOT events."""
    global _broadcast_fn
    _broadcast_fn = fn


# ── Screenshot helper ─────────────────────────────────────────────────────────

async def _broadcast_screenshot() -> None:
    """Take a viewport screenshot and send it to all connected WS clients."""
    if _broadcast_fn is None:
        return
    try:
        manager = await BrowserManager.get()
        b64 = await manager.screenshot_b64()
        page = await manager.page()
        _broadcast_fn("WEB_SCREENSHOT", {
            "image":     b64,
            "url":       page.url,
            "timestamp": time.time(),
        })
    except Exception as exc:
        print(f"  [web_agent] screenshot broadcast failed: {exc}", flush=True)


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def _web_navigate(args: dict) -> str:
    url = args.get("url", "")
    manager = await BrowserManager.get()
    page    = await manager.page()
    result  = await navigate(page, url)
    await _broadcast_screenshot()
    return result


async def _web_click(args: dict) -> str:
    selector = args.get("selector", "")
    manager  = await BrowserManager.get()
    page     = await manager.page()
    result   = await click(page, selector)
    await _broadcast_screenshot()
    return result


async def _web_type(args: dict) -> str:
    selector = args.get("selector", "")
    text     = args.get("text", "")
    manager  = await BrowserManager.get()
    page     = await manager.page()
    result   = await type_text(page, selector, text)
    await _broadcast_screenshot()
    return result


async def _web_read(args: dict) -> str:
    max_chars = int(args.get("max_chars", 8_000))
    manager   = await BrowserManager.get()
    page      = await manager.page()
    return await get_page_text(page, max_chars)  # no screenshot — read-only


async def _web_links(args: dict) -> str:
    manager = await BrowserManager.get()
    page    = await manager.page()
    return await get_page_links(page)  # no screenshot — read-only


async def _web_scroll(args: dict) -> str:
    direction = args.get("direction", "down")
    amount    = int(args.get("amount", 3))
    manager   = await BrowserManager.get()
    page      = await manager.page()
    result    = await scroll(page, direction, amount)
    await _broadcast_screenshot()
    return result


async def _web_search(args: dict) -> str:
    query   = args.get("query", "")
    manager = await BrowserManager.get()
    page    = await manager.page()
    result  = await search_google(page, query)
    await _broadcast_screenshot()
    return result


async def _web_press_key(args: dict) -> str:
    key     = args.get("key", "Enter")
    manager = await BrowserManager.get()
    page    = await manager.page()
    result  = await press_key(page, key)
    await _broadcast_screenshot()
    return result


async def _web_wait(args: dict) -> str:
    selector   = args.get("selector", "")
    timeout_ms = int(args.get("timeout_ms", 10_000))
    manager    = await BrowserManager.get()
    page       = await manager.page()
    return await wait_for(page, selector, timeout_ms)  # no screenshot — wait only


async def _web_screenshot(args: dict) -> str:
    manager = await BrowserManager.get()
    page    = await manager.page()
    await _broadcast_screenshot()
    return f"Screenshot taken — current URL: {page.url}"


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name        = "web_navigate",
    description = (
        "Navigate the browser to a URL. Use this as a first step only — "
        "always follow with web_read or web_search to get actual content. "
        "Never treat navigation alone as the final answer."
    ),
    parameters  = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to navigate to"},
        },
        "required": ["url"],
    },
    handler = _web_navigate,
))

registry.register(ToolDefinition(
    name        = "web_click",
    description = "Click an element on the current page by CSS selector or visible text",
    parameters  = {
        "type": "object",
        "properties": {
            "selector": {
                "type":        "string",
                "description": "CSS selector (e.g. '#submit-btn') or visible button/link text",
            },
        },
        "required": ["selector"],
    },
    handler = _web_click,
))

registry.register(ToolDefinition(
    name        = "web_type",
    description = "Type text into a form field on the current page",
    parameters  = {
        "type": "object",
        "properties": {
            "selector": {"type": "string", "description": "CSS selector for the input field"},
            "text":     {"type": "string", "description": "Text to type"},
        },
        "required": ["selector", "text"],
    },
    handler = _web_type,
))

registry.register(ToolDefinition(
    name        = "web_read",
    description = "Read the visible text content of the current page (up to max_chars characters)",
    parameters  = {
        "type": "object",
        "properties": {
            "max_chars": {
                "type":        "integer",
                "description": "Maximum characters to return (default 8000)",
                "default":     8000,
            },
        },
    },
    handler = _web_read,
))

registry.register(ToolDefinition(
    name        = "web_links",
    description = "Get all hyperlinks on the current page as a list",
    parameters  = {"type": "object", "properties": {}},
    handler     = _web_links,
))

registry.register(ToolDefinition(
    name        = "web_scroll",
    description = "Scroll the current page up or down",
    parameters  = {
        "type": "object",
        "properties": {
            "direction": {
                "type":        "string",
                "description": "Scroll direction: 'up' or 'down'",
                "enum":        ["up", "down"],
            },
            "amount": {
                "type":        "integer",
                "description": "Number of screen-heights to scroll (default 3)",
                "default":     3,
            },
        },
    },
    handler = _web_scroll,
))

registry.register(ToolDefinition(
    name        = "web_search",
    description = "Search Google for a query and return the top 10 results with links",
    parameters  = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
    handler = _web_search,
))

registry.register(ToolDefinition(
    name        = "web_press_key",
    description = "Press a keyboard key on the current page (e.g. Enter, Tab, Escape)",
    parameters  = {
        "type": "object",
        "properties": {
            "key": {
                "type":        "string",
                "description": "Key to press (e.g. 'Enter', 'Tab', 'Escape', 'ArrowDown')",
            },
        },
        "required": ["key"],
    },
    handler = _web_press_key,
))

registry.register(ToolDefinition(
    name        = "web_wait",
    description = "Wait for a CSS selector to appear on the current page",
    parameters  = {
        "type": "object",
        "properties": {
            "selector":   {"type": "string",  "description": "CSS selector to wait for"},
            "timeout_ms": {
                "type":        "integer",
                "description": "Maximum wait in milliseconds (default 10000)",
                "default":     10000,
            },
        },
        "required": ["selector"],
    },
    handler = _web_wait,
))

registry.register(ToolDefinition(
    name        = "web_screenshot",
    description = "Take a screenshot of the current page and send it to the frontend",
    parameters  = {"type": "object", "properties": {}},
    handler     = _web_screenshot,
))
