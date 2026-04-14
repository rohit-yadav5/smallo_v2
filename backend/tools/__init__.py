"""backend/tools — Small O tool registry and built-in tools.

Importing this package registers all four Phase-1 tools into the singleton
ToolRegistry.  The orchestrator (backend/main.py) simply does:

    from tools import file_tool, browser_tool, terminal_tool, reminder_tool

and all tools are live.  Adding a new tool in Phase 2 means:
  1. Create backend/tools/my_tool.py that calls registry.register(...) at
     module level.
  2. Add ``from tools import my_tool`` in backend/main.py.
  That's it — no changes to the registry or LLM layer needed.

Re-export the registry singleton so callers can use:
    from tools import registry
"""

from tools.registry import registry  # noqa: F401  (re-export)

# Importing each module triggers its self-registration side-effect.
from tools import file_tool          # noqa: F401
from tools import browser_tool       # noqa: F401
from tools import terminal_tool      # noqa: F401
from tools import reminder_tool      # noqa: F401
from tools import close_heavy_tabs   # noqa: F401
