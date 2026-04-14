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

# Supplemental patterns for named sites and domain patterns.
# Catches "open github" (no TLD), "find the person who made Linux on github", etc.
# Kept as a list so new entries are easy to add without modifying a complex regex.
_WEB_BYPASS_PATTERNS: list[str] = [
    r'https?://',                                                              # explicit URL
    r'\b\w+\.(com|org|net|io|dev|ai|co|uk|app)\b',                           # any domain with TLD
    r'\b(github|youtube|twitter|reddit|instagram|linkedin|google|wikipedia|'
    r'stackoverflow|hackernews|hacker\s+news|npm|pypi|gitlab|bitbucket)\b',   # named sites
]


def _is_web_query(text: str) -> bool:
    """Return True if the query references a website, URL, or well-known web service."""
    tl = text.lower()
    return any(re.search(p, tl) for p in _WEB_BYPASS_PATTERNS)


# Math / calculation patterns — LLM can answer these inline, no plugin needed.
# Any match → skip ALL plugins, route directly to LLM.
_MATH_PATTERNS: list[str] = [
    r'\d+\s*[\+\-\*\/\^%]\s*\d+',   # arithmetic: 2 + 2, 10 * 5, 50%
    r'\bwhat\s+is\s+\d',             # "what is 2...", "what is 15%..."
    r'\bcalculate\b',
    r'\bconvert\b.*\bto\b',          # unit conversions: "convert 5 km to miles"
    r'\bhow\s+much\s+is\b',
    r'\bsquare\s+root\b',
    r'\bpercent(?:age)?\s+of\b',     # "15 percent of 340"
]


def _is_math_query(text: str) -> bool:
    """Return True if the query is a pure math/calculation question the LLM can answer inline."""
    return any(re.search(p, text, re.IGNORECASE) for p in _MATH_PATTERNS)


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
          3. Math / calculation questions      → LLM (no plugin can answer these)
        """
        text = user_text.strip()

        if _CONVERSATIONAL_BYPASS.search(text):
            return None

        if _WEB_BYPASS.search(text) or _is_web_query(text):
            print("  [plugin] web bypass — routing to LLM/web-agent", flush=True)
            return None

        if _is_math_query(text):
            print("  [plugin] math query — routing to LLM", flush=True)
            return None

        for plugin in self._plugins:
            # Word-count guard: the computer plugin only handles short, specific
            # OS-level open commands ("open Finder", "open Safari").  Queries
            # longer than 4 words — e.g. "open github and find the person who
            # made Linux" — must never reach computer.open_target.
            if getattr(plugin, "NAME", "") == "computer" and len(text.split()) > 4:
                continue
            result = plugin.match(user_text)
            if result is not None:
                action, match_obj = result
                print(f"  [plugin] Matched → {plugin.NAME}.{action}")
                return plugin.execute(user_text, action, match_obj)
        return None
