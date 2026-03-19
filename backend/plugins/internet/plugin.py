import re
from datetime import datetime
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from plugins.base import BasePlugin, PluginResult

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SmallO/2.0)"}


class InternetPlugin(BasePlugin):
    NAME = "internet"
    PRIORITY = 20

    INTENTS = [
        # Time / date — must come before broad "what is" wikipedia pattern
        (r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?time\b", "current_time"),
        (r"\bwhat\s+time\s+is\s+it\b", "current_time"),
        (r"\bwhat(?:'s|\s+is)\s+(?:today'?s?\s+)?date\b", "current_time"),
        (r"\bwhat\s+(?:is\s+)?today\s+date\b", "current_time"),
        (r"\bwhat\s+(?:day|month|year)\s+is\s+it\b", "current_time"),
        # Web search
        (r"\bsearch\s+(?:for\s+|the\s+web\s+for\s+)?(?P<query>.+)", "web_search"),
        (r"\bgoogle\s+(?P<query>.+)", "web_search"),
        (r"\blook\s+up\s+(?P<query>.+)", "web_search"),
        (r"\bfind\s+(?:information\s+)?(?:about|on)\s+(?P<query>.+)", "web_search"),
        # Weather
        (r"\bweather\b", "weather"),
        (r"\btemperature\b", "weather"),
        (r"\bwhat.s\s+it\s+like\s+outside\b", "weather"),
        # URL fetch
        (r"\bfetch\b.+(?P<url>https?://\S+)", "fetch_url"),
        (r"\bsummarize\b.+(?P<url>https?://\S+)", "fetch_url"),
        # Wikipedia — exclude personal/demonstrative pronouns to avoid swallowing
        # conversational questions like "what is your name" or "who is she"
        (r"\bwikipedia\s+(?P<topic>.+)", "wikipedia"),
        (r"\btell\s+me\s+about\s+(?P<topic>(?!(?:you|your|me|my|us|our|him|her|them|their|this|that|it|a\b|an\b)\b).+)", "wikipedia"),
        (r"\bwho\s+is\s+(?P<topic>(?!(?:you|he|she|they|it)\b).+)", "wikipedia"),
        (r"\bwhat\s+is\s+(?P<topic>(?!(?:your|my|his|her|our|their|its|it\b|this|that|a\b|an\b|the\b)\b).+)", "wikipedia"),
    ]

    def execute(self, user_text: str, action: str, match: re.Match) -> PluginResult:
        handler = getattr(self, f"_{action}", None)
        if handler is None:
            return {"text": f"Unknown internet action: {action}", "direct": True,
                    "plugin": self.NAME, "action": action}
        try:
            return handler(user_text, match)
        except Exception as e:
            return {"text": f"Internet plugin error: {e}", "direct": True,
                    "plugin": self.NAME, "action": action}

    # ── Actions ──────────────────────────────────────────────────────────────

    def _web_search(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            query = match.group("query").strip()
        except IndexError:
            query = user_text
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        snippets = []
        for result in soup.select(".result__snippet")[:3]:
            text = result.get_text(strip=True)
            if text:
                snippets.append(text)
        if not snippets:
            return {"text": f"No results found for: {query}", "direct": True,
                    "plugin": self.NAME, "action": "web_search"}
        combined = f"Search results for '{query}':\n" + "\n".join(f"- {s}" for s in snippets)
        return {"text": combined, "direct": False, "plugin": self.NAME, "action": "web_search"}

    def _weather(self, user_text: str, match: re.Match) -> PluginResult:
        # Try to extract a city name from the user's text
        city_match = re.search(r"(?:in|for|at)\s+([A-Za-z\s]+?)(?:\?|$)", user_text, re.IGNORECASE)
        city = city_match.group(1).strip() if city_match else ""
        target = quote(city) if city else ""
        url = f"https://wttr.in/{target}?format=3"
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        text = resp.text.strip()
        if not text or "Unknown location" in text:
            text = "Could not get weather. Try saying 'weather in London'."
            return {"text": text, "direct": True, "plugin": self.NAME, "action": "weather"}
        return {"text": text, "direct": False, "plugin": self.NAME, "action": "weather"}

    def _wikipedia(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            topic = match.group("topic").strip()
        except IndexError:
            topic = user_text
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(topic)}"
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        if resp.status_code != 200:
            return {"text": f"No Wikipedia article found for: {topic}", "direct": True,
                    "plugin": self.NAME, "action": "wikipedia"}
        data = resp.json()
        extract = data.get("extract", "No summary available.")
        # Truncate to first 400 chars so TTS isn't too long
        return {"text": extract[:400], "direct": False, "plugin": self.NAME, "action": "wikipedia"}

    def _fetch_url(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            url = match.group("url").strip()
        except IndexError:
            return {"text": "No URL found. Please include a full URL starting with http.",
                    "direct": True, "plugin": self.NAME, "action": "fetch_url"}
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # First 500 chars of readable content
        return {"text": text[:500], "direct": False, "plugin": self.NAME, "action": "fetch_url"}

    def _current_time(self, user_text: str, match: re.Match) -> PluginResult:
        now = datetime.now()
        time_str = now.strftime("It's %I:%M %p on %A, %B %d, %Y.")
        return {"text": time_str, "direct": True, "plugin": self.NAME, "action": "current_time"}
