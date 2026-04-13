import importlib
import re
from pathlib import Path

from plugins.base import BasePlugin, PluginResult

# Questions that should always go straight to the LLM, never to a plugin.
# Matches conversational/personal queries like "what is your name",
# "can you help me", "how are you", "what do you think", etc.
_CONVERSATIONAL_BYPASS = re.compile(
    r"^(?:"
    # "what/who/how/where/when ... you/your/i/my/we/our ..."
    r"(?:what|who|where|when|how)\s+(?:is|are|do|did|can|could|would|will|should|was|were)\s+(?:you|your|i|my|we|our|the\s+(?:time|date|day))\b"
    r"|what(?:'s|\s+is)\s+(?:your|my|his|her|our|its)\b"  # what is your/my ...
    r"|(?:can|could|would|will|should|do|does|did|are|is)\s+you\b"  # can you, are you
    r"|(?:tell\s+me\s+(?:about\s+)?)?(?:your|yourself)\b"           # tell me about yourself
    r"|(?:hello|hi|hey|thanks|thank you|bye|goodbye|good\s+(?:morning|afternoon|evening|night))\b"
    r")",
    re.IGNORECASE,
)

# Web/browser-related inputs that should fall through to the LLM so the
# web agent tools (web_navigate, web_search, deep_research, …) can handle them.
# Any match → router returns None without consulting any plugin.
_WEB_BYPASS = re.compile(
    r"""
    \bin\s+browser\b                    # "open X in browser"
    | open\s+\S+\.(?:com|org|net|io|co|app|dev|ai)\b  # "open instagram.com"
    | (?:^|\s)go\s+to\s                 # "go to ..."
    | (?:^|\s)navigate\s+to\s          # "navigate to ..."
    | (?:^|\s)search\s+google          # "search google for ..."
    | (?:^|\s)fetch\s+http             # "fetch https://..."
    | (?:^|\s)web_                     # raw tool names like "web_navigate"
    | \bwebsite\b                      # "open the website"
    | \bwebpage\b                      # "load the webpage"
    | \bbrowse\s+to\b                  # "browse to ..."
    | \burl\b                          # "open this url"
    """,
    re.IGNORECASE | re.VERBOSE,
)


class PluginRouter:
    """
    Auto-discovers all plugins/*/plugin.py files, instantiates each plugin,
    and routes user_text to the first matching plugin by PRIORITY order.
    """

    def __init__(self):
        self._plugins: list[BasePlugin] = []
        self._load_plugins()

    def _load_plugins(self):
        plugins_dir = Path(__file__).parent
        for plugin_file in sorted(plugins_dir.glob("*/plugin.py")):
            module_path = f"plugins.{plugin_file.parent.name}.plugin"
            try:
                module = importlib.import_module(module_path)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BasePlugin)
                        and attr is not BasePlugin
                    ):
                        self._plugins.append(attr())
                        break
            except Exception as e:
                print(f"  [plugin] Failed to load {module_path}: {e}")

        self._plugins.sort(key=lambda p: p.PRIORITY)
        names = [p.NAME for p in self._plugins]
        print(f"  [plugin] Loaded {len(self._plugins)} plugin(s): {names}")

    def route(self, user_text: str) -> PluginResult | None:
        """
        Try each plugin in priority order.
        Returns a PluginResult on first match, else None.

        Bypass order (checked before any plugin):
          1. Conversational/personal questions → LLM
          2. Web/browser-related commands      → LLM → web agent tools
        """
        text = user_text.strip()

        if _CONVERSATIONAL_BYPASS.search(text):
            return None

        if _WEB_BYPASS.search(text):
            print("  [plugin] web bypass — routing to LLM/web-agent", flush=True)
            return None

        for plugin in self._plugins:
            result = plugin.match(user_text)
            if result is not None:
                action, match_obj = result
                print(f"  [plugin] Matched → {plugin.NAME}.{action}")
                return plugin.execute(user_text, action, match_obj)
        return None
