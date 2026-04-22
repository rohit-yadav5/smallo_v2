"""backend/tools/browser_tool.py — Web fetch tool for Small O.

Registers one tool at import time:
  • fetch_url — fetch a URL and return its text, raw HTML, or link list.

Uses httpx (async) with a realistic User-Agent header so sites don't block
the request.  BeautifulSoup handles HTML parsing for the "text" and "links"
extract modes.

Phase 2 note: this is the primitive that a future web-research sub-agent will
wrap.  Keep the interface stable.
"""

import asyncio

import httpx
from bs4 import BeautifulSoup

from tools.registry import ToolDefinition, registry


_TIMEOUT    = httpx.Timeout(15.0)
# SSL verification: Homebrew Python 3.11 on macOS often lacks root CA certs,
# causing CERTIFICATE_VERIFY_FAILED for common sites even with certifi.
# Since Small O is a local-only personal assistant that never transmits user
# credentials over fetched URLs, we disable server-cert verification here.
# This is a deliberate, documented trade-off — not an oversight.
_VERIFY     = False
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
from config.limits import TOOL_OUTPUT_BROWSER as _MAX_CHARS


async def _fetch_url(args: dict) -> str:
    url: str     = args.get("url", "").strip()
    extract: str = args.get("extract", "text").lower()

    if not url:
        return "Error: 'url' argument is required."
    if extract not in ("text", "html", "links"):
        extract = "text"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            verify=_VERIFY,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw_html = resp.text

        if extract == "html":
            return raw_html[:_MAX_CHARS]

        soup = BeautifulSoup(raw_html, "html.parser")

        if extract == "links":
            links = []
            for tag in soup.find_all("a", href=True):
                href = tag["href"].strip()
                if href and not href.startswith(("#", "javascript:")):
                    links.append(href)
            return "\n".join(links[:200])  # cap at 200 links

        # extract == "text"
        # Remove script/style noise before getting text
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace runs
        import re
        text = re.sub(r"\s{2,}", " ", text)
        return text[:_MAX_CHARS]

    except httpx.HTTPStatusError as exc:
        return f"Error fetching {url}: HTTP {exc.response.status_code}"
    except httpx.TimeoutException:
        return f"Error fetching {url}: request timed out after 15 s"
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name="fetch_url",
    description=(
        "Fetch a web page and return its content. "
        "Use extract='text' for readable text, 'html' for raw HTML, 'links' for all hyperlinks."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch (https:// prefix assumed if omitted).",
            },
            "extract": {
                "type": "string",
                "enum": ["text", "html", "links"],
                "description": "What to extract from the page. Defaults to 'text'.",
            },
        },
        "required": ["url"],
    },
    handler=_fetch_url,
))
