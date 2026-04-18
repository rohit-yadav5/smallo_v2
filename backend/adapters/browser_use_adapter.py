"""backend/adapters/browser_use_adapter.py

Wraps browser-use 0.12.6 to add a high-level `web_task` tool.

The existing low-level web_* tools (web_navigate, web_click, etc.) remain
untouched.  This adapter adds a single autonomous `web_task` tool that lets
the LLM describe a browser goal in plain English and have browser-use execute
it step by step.

Configuration
─────────────
browser-use 0.12.6 has its own LLM class at browser_use.llm.openai.chat.
We use that directly (not langchain_openai) so the .model and .provider
attributes that browser-use checks internally are always present.

Step callbacks
──────────────
Agent.run() accepts `on_step_end: async (Agent) -> None`.  We hook into this
to emit WEB_SCREENSHOT events after each browser step, giving the frontend
a live view of what the agent is doing.
"""

import asyncio
import base64
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
    """Return a browser-use ChatOpenAI instance pointed at local Ollama (no cloud calls).

    Uses browser_use.llm.openai.chat.ChatOpenAI (not langchain_openai) so that
    the .model and .provider attributes browser-use checks internally are present.
    Lazy import so tests can mock the module before it is resolved.
    """
    from browser_use.llm.openai.chat import ChatOpenAI as BrowserUseChatOpenAI  # noqa: PLC0415
    return BrowserUseChatOpenAI(
        model=LLM_CONFIG.planner_model,
        base_url="http://localhost:11434/v1",
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
    task: str = args.get("task") or args.get("goal", "")
    if not task:
        return "Error: task (or goal) is required."

    max_steps: int = int(args.get("max_steps", 20))

    print(f"  [browser_use] starting web_task: {task!r}  max_steps={max_steps}", flush=True)

    try:
        llm = _make_llm()

        # Step callback: emits a screenshot after each agent step
        async def _on_step_end(agent) -> None:
            await _try_broadcast_screenshot(agent)

        from browser_use import Agent  # noqa: PLC0415 — function-level to allow test mocking
        agent = Agent(
            task=task,
            llm=llm,
            enable_signal_handler=False,
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
        "CALL THIS DIRECTLY — do NOT use the planner. "
        "Autonomously drives a real browser to complete a plain-English goal: "
        "navigates pages, clicks buttons, fills forms, extracts content. "
        "Use when the task needs real browser interaction (prices, logins, dynamic pages). "
        "Accepts 'task' or 'goal' as the parameter name."
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
