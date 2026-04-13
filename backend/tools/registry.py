"""backend/tools/registry.py — Central tool registry for Small O's Jarvis upgrade.

The ToolRegistry is a module-level singleton.  Each tool file calls
``registry.register(ToolDefinition(...))`` at import time so that simply
importing a tool module is enough to make it available to the LLM.

Design principles
─────────────────
• Self-registering: tools declare themselves; the orchestrator just imports.
• Async-first: every handler is an async callable so tools can do I/O without
  blocking the event loop.
• Schema-driven: each tool carries a JSON Schema ``parameters`` dict that gets
  injected verbatim into the LLM system prompt, so the model always has an
  accurate spec.
• Phase-2-ready: dispatch() is async and returns a plain str, making it trivial
  for a future task-planner to call multiple tools in a loop without coupling to
  any particular LLM framework.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """Describes a single callable tool.

    Parameters
    ----------
    name:        snake_case identifier used in <tool_call> JSON blocks.
    description: one-sentence summary injected into the LLM prompt.
    parameters:  JSON Schema ``object`` describing the args dict.
    handler:     async (args: dict) -> str  — must never raise; return errors
                 as plain strings so the LLM can reason about them.
    """
    name: str
    description: str
    parameters: dict          # JSON Schema
    handler: Callable         # async (args: dict) -> str


class _ToolRegistry:
    """Singleton registry for all Small O tools.

    Usage
    ─────
    from tools.registry import registry

    # Register (done inside each tool module at import time):
    registry.register(ToolDefinition(name="my_tool", ...))

    # Query (done inside llm/main_llm.py before each LLM call):
    schemas = registry.get_schemas()

    # Execute (done inside llm/main_llm.py after tool-call detection):
    result = await registry.dispatch("my_tool", {"arg": "value"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool.  Overwrites any existing tool with the same name."""
        if not asyncio.iscoroutinefunction(tool.handler):
            raise TypeError(
                f"Tool '{tool.name}' handler must be an async function (async def)."
            )
        self._tools[tool.name] = tool
        print(f"  [tools] registered: {tool.name}", flush=True)

    # ── Query ────────────────────────────────────────────────────────────────

    def get_schemas(self) -> list[dict]:
        """Return a list of JSON-Schema-style tool descriptors for LLM injection."""
        return [
            {
                "name":        t.name,
                "description": t.description,
                "parameters":  t.parameters,
            }
            for t in self._tools.values()
        ]

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    # ── Dispatch ─────────────────────────────────────────────────────────────

    async def dispatch(self, name: str, args: dict) -> str:
        """Call a registered tool by name.

        Never raises — all errors are returned as a string so the LLM can
        incorporate them into its next response naturally.
        """
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(self._tools.keys()) or "none"
            return f"Error: unknown tool '{name}'. Known tools: {known}"
        try:
            result = await tool.handler(args)
            return str(result)
        except Exception as exc:
            return f"Error running tool '{name}': {exc}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ToolRegistry tools={list(self._tools.keys())}>"


# Module-level singleton — import this everywhere.
registry = _ToolRegistry()
