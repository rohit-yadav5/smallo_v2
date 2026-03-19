import re
from abc import ABC, abstractmethod
from typing import TypedDict


class PluginResult(TypedDict):
    text: str      # Content to speak or pass to LLM for summarization
    direct: bool   # True = speak as-is; False = pass through LLM first
    plugin: str    # Plugin name, e.g. "computer"
    action: str    # Action taken, e.g. "open_target"


class BasePlugin(ABC):
    """
    Base class for all plugins.

    Subclasses declare INTENTS as a list of (regex_pattern, action_name) tuples.
    Patterns are compiled once at class definition time via __init_subclass__.
    The router calls match() to detect intent, then execute() to run the action.
    """

    NAME: str = ""
    PRIORITY: int = 50  # Lower = checked first by router

    # Each entry: (regex_pattern_string, action_name)
    # Patterns use re.IGNORECASE and re.search (not re.match)
    INTENTS: list[tuple[str, str]] = []

    # Populated by __init_subclass__ — compiled patterns per subclass
    _compiled: list[tuple[re.Pattern, str]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._compiled = [
            (re.compile(pat, re.IGNORECASE), action)
            for pat, action in cls.INTENTS
        ]

    def match(self, user_text: str) -> tuple[str, re.Match] | None:
        """
        Return (action_name, match_obj) for the first matching intent, or None.
        Called by the router.
        """
        for pattern, action in self._compiled:
            m = pattern.search(user_text)
            if m:
                return action, m
        return None

    @abstractmethod
    def execute(self, user_text: str, action: str, match: re.Match) -> PluginResult:
        """
        Perform the action and return a PluginResult.
        Must never raise — catch all exceptions and return an error PluginResult.
        """
        ...
