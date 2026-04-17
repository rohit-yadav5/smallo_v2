"""backend/adapters/browser_use_adapter.py

Wraps browser-use 0.12.6 to add a high-level `web_task` tool.

The existing low-level web_* tools (web_navigate, web_click, etc.) remain
untouched.  This adapter adds a single autonomous `web_task` tool that lets
the LLM describe a browser goal in plain English and have browser-use execute
it step by step.

Configuration
─────────────
browser-use uses LangChain-compatible LLMs.  We pass a ChatOpenAI instance
pointed at the local Ollama OpenAI-compatible endpoint so that zero cloud
calls are made.

Step callbacks
──────────────
Agent.run() accepts `on_step_end: async (Agent) -> None`.  We hook into this
to emit WEB_SCREENSHOT events after each browser step, giving the frontend
a live view of what the agent is doing.
"""

import asyncio
import base64
import importlib
import time
from typing import Callable, Optional

from config.llm import LLM_CONFIG
from tools.registry import registry, ToolDefinition


# ── Broadcast hook ────────────────────────────────────────────────────────────

_broadcast_fn: Optional[Callable] = None


def set_broadcast_fn(fn: Callable) -> None:
    """Register the _emit callback from main.py for WEB_SCREENSHOT events.

    Expected signature: async (event_type: str, payload: dict) -> None
    """
    global _broadcast_fn
    _broadcast_fn = fn


# ── LLM factory ───────────────────────────────────────────────────────────────

def _make_llm():
    """Return a ChatOpenAI instance pointed at local Ollama (no cloud calls).

    Lazy import so tests can mock langchain_openai without the module-level
    import chain pulling in torch on first load.
    """
    from langchain_openai import ChatOpenAI  # noqa: PLC0415 — intentional lazy import
    return ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model=LLM_CONFIG.planner_model,
        api_key="ollama",
        temperature=0,
    )


# ── Screenshot helper ─────────────────────────────────────────────────────────

async def _try_broadcast_screenshot(agent) -> None:
    """Best-effort: grab a screenshot from the agent's browser and broadcast it.

    Silently swallows all exceptions — screenshots are non-fatal.
    """
    if _broadcast_fn is None:
        return
    try:
        # browser-use Agent exposes browser_session.get_current_page()
        browser_session = getattr(agent, "browser_session", None)
        if browser_session is None:
            return

        page = None
        if hasattr(browser_session, "get_current_page"):
            page = await browser_session.get_current_page()
        elif hasattr(browser_session, "page"):
            page = browser_session.page

        if page is None:
            return

        screenshot_bytes = await page.screenshot(type="jpeg", quality=60)
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        url = page.url

        payload = {"image": b64, "url": url, "timestamp": time.time()}
        if asyncio.iscoroutinefunction(_broadcast_fn):
            await _broadcast_fn("WEB_SCREENSHOT", payload)
        else:
            _broadcast_fn("WEB_SCREENSHOT", payload)
    except Exception as exc:
        print(f"  [browser_use_adapter] screenshot broadcast skipped: {exc}", flush=True)


# ── Tool handler ──────────────────────────────────────────────────────────────

async def _web_task(args: dict) -> str:
    """Async handler registered in ToolRegistry for the 'web_task' tool.

    Accepts:
        task      (str, required) — plain-English description of the browser goal
        max_steps (int, optional) — max number of agent steps (default 20)
    """
    task: str = args.get("task", "")
    if not task:
        return "Error: task is required."

    max_steps: int = int(args.get("max_steps", 20))

    print(f"  [browser_use] starting web_task: {task!r}  max_steps={max_steps}", flush=True)

    try:
        llm = _make_llm()

        # Step callback: emits a screenshot after each agent step
        async def _on_step_end(agent) -> None:
            await _try_broadcast_screenshot(agent)

        # Defer the import so tests can patch sys.modules["browser_use"] before use
        _browser_use = importlib.import_module("browser_use")
        Agent = _browser_use.Agent
        agent = Agent(
            task=task,
            llm=llm,
        )

        history = await agent.run(
            max_steps=max_steps,
            on_step_end=_on_step_end,
        )

        final = history.final_result()
        if final:
            return str(final)

        # Fallback: stringify the history object if final_result() returned None
        return f"web_task completed ({history.number_of_steps()} steps)."

    except Exception as exc:
        err_msg = f"web_task failed: {exc}"
        print(f"  [browser_use_adapter] {err_msg}", flush=True)
        return err_msg


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name="web_task",
    description=(
        "Autonomously complete a browser-based task using browser-use (local Ollama, "
        "no cloud calls): navigates, clicks, types, and reads web pages to achieve the "
        "described goal. Use for multi-step browser tasks that require real interaction "
        "with web pages (e.g. filling forms, scraping dynamic content, web automation)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Plain-English description of the browser goal to achieve",
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum number of agent steps to take (default 20)",
                "default": 20,
            },
        },
        "required": ["task"],
    },
    handler=_web_task,
))
