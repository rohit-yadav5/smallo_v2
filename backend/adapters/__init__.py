"""backend/adapters — wrappers for external AI libraries (GPT-Researcher, browser-use).

Each adapter:
  - Configures its library to use local Ollama (zero cloud calls)
  - Self-registers its tool(s) in ToolRegistry at import time
  - Exposes set_broadcast_fn() for WS event emission
"""
