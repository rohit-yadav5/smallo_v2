import re
from datetime import datetime
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from plugins.base import BasePlugin, PluginResult

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SmallO/2.0)"}

# Strip non-ASCII characters that TTS can't speak (emoji, ideographs, etc.).
# Keeps standard Latin, extended Latin, and common punctuation/dashes.
_NON_SPEAKABLE_RE = re.compile(r'[^\x00-\x7F\u00C0-\u024F\u2010-\u2027]')


def _clean_for_tts(text: str) -> str:
    """Remove emoji and other non-speakable Unicode before text goes to TTS."""
    return _NON_SPEAKABLE_RE.sub('', text).strip()


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

    # Arithmetic operator pattern — used to guard the wikipedia intent.
    _OPERATOR_RE = re.compile(r'[\+\*\/=]|\b\d+\s*-\s*\d+\b', re.IGNORECASE)

    def match(self, user_text: str) -> tuple[str, re.Match] | None:
        """
        Extend BasePlugin.match with guards for the wikipedia action:
          • Query must be ≥ 4 words (avoids swallowing "what is 2 + 2")
          • Query must not contain arithmetic operators (+, *, /, =, N-N)
        If the wikipedia pattern would match but the guards fail, skip it and
        let the LLM answer directly.
        """
        for pattern, action in self._compiled:
            m = pattern.search(user_text)
            if m:
                if action == "wikipedia":
                    word_count = len(user_text.split())
                    if word_count < 4 or self._OPERATOR_RE.search(user_text):
                        continue   # guards failed — skip this match
                return action, m
        return None

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
        # Force UTF-8 decode — requests sometimes misdetects encoding on wttr.in
        # responses, producing mangled characters like "ð«" instead of weather emoji.
        text = resp.content.decode("utf-8", errors="replace").strip()
        # Strip emoji and non-speakable characters so TTS doesn't attempt to
        # read "thunderstorm emoji" aloud, which sounds terrible.
        text = _clean_for_tts(text)
        if not text or "Unknown location" in text:
            text = "Could not get weather. Try saying 'weather in London'."
            return {"text": text, "direct": True, "plugin": self.NAME, "action": "weather"}
        return {"text": text, "direct": False, "plugin": self.NAME, "action": "weather"}

    @staticmethod
    def _extract_wiki_topic(raw_query: str) -> str:
        """Convert a free-form question into the best Wikipedia article title.

        Strategy:
        1. Strip leading question words and filler so we get a clean noun phrase.
        2. Call the Wikipedia opensearch API to find the closest article title.
        3. Fall back to the stripped noun phrase if the API call fails.
        """
        # Strip trailing punctuation
        cleaned = raw_query.strip().rstrip("?.!").strip()

        # Strip leading question/filler prefixes iteratively until stable
        prefixes = [
            r"^tell\s+me\s+(?:about|of)\s+",
            r"^who\s+(?:is|was|are|were)\s+",
            r"^what\s+(?:is|was|are|were)\s+",
            r"^where\s+(?:is|was|are|were)\s+",
            r"^when\s+(?:is|was|are|were)\s+",
            r"^which\s+(?:is|was|are|were)\s+",
            r"^how\s+(?:is|was|are|were)\s+",
            r"^(?:the|a|an)\s+",  # strip leading articles last
        ]
        prev = None
        while prev != cleaned:
            prev = cleaned
            for pat in prefixes:
                cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

        cleaned = cleaned.strip()

        # Use Wikipedia opensearch to resolve the best article title
        try:
            search_url = (
                "https://en.wikipedia.org/w/api.php"
                "?action=opensearch&namespace=0&limit=1&format=json"
                f"&search={quote(cleaned)}"
            )
            resp = requests.get(search_url, headers=_HEADERS, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                # data[1] is a list of matching titles
                if data and len(data) > 1 and data[1]:
                    return data[1][0]  # first (best) matching article title
        except Exception:
            pass

        return cleaned  # fallback: use cleaned noun phrase directly

    def _wikipedia(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            raw_topic = match.group("topic").strip()
        except IndexError:
            raw_topic = user_text

        # Resolve the raw regex capture to a proper Wikipedia article title
        topic = self._extract_wiki_topic(raw_topic)

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
