"""backend/web_agent — Playwright-powered persistent browser agent for Small O.

Importing this package triggers self-registration of all web_* tools into the
shared ToolRegistry so both normal LLM turns and the planner can use them.

Sub-modules
───────────
  browser   — BrowserManager singleton (persistent Chromium, headed)
  actions   — Raw async Playwright helpers (navigate, click, type, …)
  agent     — ToolRegistry integration + screenshot broadcast loop
  monitor   — Background webpage-change detector (PROACTIVE_EVENT on change)
"""

# Importing agent triggers tool self-registration at module load time.
from web_agent import agent  # noqa: F401
from web_agent.monitor import web_monitor  # noqa: F401
