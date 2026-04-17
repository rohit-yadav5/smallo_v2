"""backend/adapters/gpt_researcher_adapter.py

Wraps gpt-researcher 0.14.8 to replace backend/tools/research_tool.py.

Configuration
─────────────
GPT-Researcher reads config from environment variables (via its Config class).
We set them with setdefault() BEFORE importing GPTResearcher so we never
stomp on variables the user may have already set.

The FAST_LLM / SMART_LLM env vars use the format "provider:model", where the
first colon separates the provider name from everything after (the model name
may itself contain colons, e.g. "openai:qwen2.5:7b").

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1.
We tell GPT-Researcher to use "openai" as the provider and point OPENAI_BASE_URL
at the local Ollama instance.  OPENAI_API_KEY is set to the dummy value "ollama"
that Ollama accepts without validation.
"""

import asyncio
import os
from typing import Callable, Optional

# ── Env vars must be set BEFORE importing GPTResearcher ─────────────────────
# LLM_CONFIG is a frozen dataclass — safe to import first.
from config.llm import LLM_CONFIG

_OLLAMA_V1_BASE = "http://localhost:11434/v1"

os.environ.setdefault("LLM_PROVIDER",     "openai")          # use OpenAI-compat layer
os.environ.setdefault("OPENAI_API_KEY",   "ollama")           # dummy key accepted by Ollama
os.environ.setdefault("OPENAI_BASE_URL",  _OLLAMA_V1_BASE)
# Format: "provider:model" — Config.parse_llm splits on first colon
os.environ.setdefault("FAST_LLM",         f"openai:{LLM_CONFIG.planner_model}")
os.environ.setdefault("SMART_LLM",        f"openai:{LLM_CONFIG.planner_model}")
os.environ.setdefault("STRATEGIC_LLM",    f"openai:{LLM_CONFIG.planner_model}")
os.environ.setdefault("RETRIEVER",        "duckduckgo")
os.environ.setdefault("DOC_PATH",         "")                 # no local doc path needed
os.environ.setdefault("EMBEDDING",        "custom:qwen2.5:3b")  # local Ollama via OpenAI-compat /v1

# ── Now safe to import GPTResearcher ────────────────────────────────────────
from gpt_researcher import GPTResearcher  # noqa: E402

from tools.registry import registry, ToolDefinition  # noqa: E402


# ── Broadcast shim (PLAN_EVENT) ──────────────────────────────────────────────

_broadcast_fn: Optional[Callable] = None


def set_broadcast_fn(fn: Callable) -> None:
    """Register a callable that broadcasts WebSocket events to all clients.

    Expected signature: async (event_type: str, payload: dict) -> None
    """
    global _broadcast_fn
    _broadcast_fn = fn


async def _emit(phase: str, message: str, topic: str = "") -> None:
    """Emit a PLAN_EVENT progress update if a broadcast function is registered."""
    if _broadcast_fn is None:
        return
    try:
        payload = {"phase": phase, "message": message, "tool": "deep_research", "topic": topic}
        if asyncio.iscoroutinefunction(_broadcast_fn):
            await _broadcast_fn("PLAN_EVENT", payload)
        else:
            _broadcast_fn("PLAN_EVENT", payload)
    except Exception as exc:
        print(f"  [gpt_researcher_adapter] broadcast error: {exc}", flush=True)


# ── Tool handler ─────────────────────────────────────────────────────────────

async def _deep_research(args: dict) -> str:
    """Async handler registered in ToolRegistry for the 'deep_research' tool.

    Accepts:
        topic    (str, required) — the research question / topic
        query    (str)           — alias for topic (backward compat)
        max_pages (int)          — passed to GPTResearcher as MAX_SEARCH_RESULTS_PER_QUERY
                                   (default 8, capped at 15)
        save_to  (str)           — optional file path; if set, also dispatches write_file
    """
    # ── Arg extraction ────────────────────────────────────────────────────────
    topic: str = args.get("topic") or args.get("query", "")
    if not topic:
        return "Error: topic (or query) is required."

    max_pages: int = min(int(args.get("max_pages", 8)), 15)
    save_to: Optional[str] = args.get("save_to") or args.get("save_path")

    print(f"  [gpt_researcher] starting research: {topic!r}  max_pages={max_pages}", flush=True)
    await _emit("start", f"Starting research on: {topic}", topic=topic)

    # ── Run GPT-Researcher ────────────────────────────────────────────────────
    try:
        await _emit("searching", "Searching and gathering sources…", topic=topic)

        # Forward max_pages to GPTResearcher via its env-var config knob.
        # Safe in a single asyncio event loop — no concurrent calls can race here.
        per_query = str(max(2, max_pages // 2))
        os.environ["MAX_SEARCH_RESULTS_PER_QUERY"] = per_query

        researcher = GPTResearcher(
            query=topic,
            report_type="research_report",
            report_source="web",
            verbose=False,
        )

        await researcher.conduct_research()

        await _emit("writing", "Writing report…", topic=topic)
        report: str = await researcher.write_report()

    except Exception as exc:
        err_msg = f"Research failed: {exc}"
        print(f"  [gpt_researcher] {err_msg}", flush=True)
        await _emit("error", err_msg, topic=topic)
        return err_msg

    # ── Optional file save ───────────────────────────────────────────────────
    if save_to:
        try:
            save_result = await registry.dispatch(
                "write_file",
                {"path": save_to, "content": report, "mode": "overwrite"},
            )
            print(f"  [gpt_researcher] saved: {save_result}", flush=True)
            await _emit("done", f"Research complete. Report saved to {save_to}", topic=topic)
            summary_preview = report[:500].replace("\n", " ")
            return (
                f"Research complete. Report saved to {save_to}\n\n"
                f"Summary: {summary_preview}…"
            )
        except Exception as exc:
            report += f"\n\n[Warning: could not save report — {exc}]"

    await _emit("done", "Research complete.", topic=topic)
    return report


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name="deep_research",
    description=(
        "Perform in-depth web research on a topic using GPT-Researcher (local Ollama, "
        "no cloud calls): searches the web, scrapes sources, and synthesizes a structured "
        "Markdown report. Use for thorough research requiring multiple web sources. "
        "Optionally saves the report to a file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The research topic or question to investigate",
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum search results to use (default 8, max 15)",
                "default": 8,
            },
            "save_to": {
                "type": "string",
                "description": "Optional file path to save the report (e.g. ~/Desktop/report.md)",
            },
        },
        "required": ["topic"],
    },
    handler=_deep_research,
))
