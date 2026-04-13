"""backend/tools/research_tool.py — Deep web research tool.

Performs multi-page research:
  1. DuckDuckGo search (falls back to Google) for the query
  2. Filter noise/social URLs, visit top N seed pages
  3. Follow one level of relevant internal links per page (depth-2 crawl)
  4. Synthesize everything with a local LLM call via llm.main_llm.ask_llm
  5. Optionally save the report to a file

The tool is registered as "deep_research" in the ToolRegistry.

Constraints
───────────
  • Uses the shared BrowserManager (same Playwright browser the user sees)
  • Caps at 5 seed URLs and 3 depth-2 links per seed to keep run time < 60s
  • Each page capped at 4 000 chars; synthesis prompt capped at 12 000 chars
  • If save_path provided, writes a markdown report and confirms filename
"""

import asyncio
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote as _url_quote, urlparse

from web_agent.browser import BrowserManager
from web_agent.actions import navigate, get_page_text, get_page_links
from tools.registry import registry, ToolDefinition


# ── URL noise filter ──────────────────────────────────────────────────────────

_NOISE_PATTERNS = (
    "youtube.com", "youtu.be",
    "google.com/products", "google.com/intl",
    "accounts.google.com", "play.google.com", "support.google.com",
    "facebook.com", "instagram.com",
    "twitter.com", "/x.com/",
    "tiktok.com",
)


def _is_noise_url(url: str) -> bool:
    """Return True if the URL matches a known low-quality / social-media domain."""
    lower = url.lower()
    return any(pat in lower for pat in _NOISE_PATTERNS)


# ── Crawl helpers ─────────────────────────────────────────────────────────────

async def _page_text(page, url: str, max_chars: int = 4_000) -> str:
    """Navigate to url, return capped page text."""
    result = await navigate(page, url)
    if "failed" in result.lower():
        return f"[failed to load {url}]"
    return await get_page_text(page, max_chars)


async def _seed_urls(page, query: str, n: int = 5) -> list[str]:
    """
    Search for query and return up to n clean seed URLs.

    Primary:  DuckDuckGo HTML (no JS required, no bot detection)
    Fallback: Google with standard query string
    """
    encoded = _url_quote(query)
    urls: list[str] = []

    # ── DuckDuckGo (primary) ──────────────────────────────────────────────
    ddg_url = f"https://duckduckgo.com/html/?q={encoded}"
    try:
        await page.goto(ddg_url, wait_until="domcontentloaded", timeout=20_000)
        urls = await page.evaluate("""
            () => Array.from(document.querySelectorAll('.result__a'))
                       .map(a => a.href)
                       .filter(h => h && h.startsWith('http'))
                       .slice(0, 15)
        """)
        print(f"  [research] DuckDuckGo returned {len(urls)} raw URL(s)", flush=True)
    except Exception as exc:
        print(f"  [research] DuckDuckGo failed ({exc}) — falling back to Google", flush=True)

    # ── Google fallback ───────────────────────────────────────────────────
    if not urls:
        google_url = f"https://www.google.com/search?q={encoded}&num=10"
        try:
            await page.goto(google_url, wait_until="domcontentloaded", timeout=20_000)
            urls = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href^="http"]'))
                           .map(a => a.href)
                           .filter(h => !h.includes('google.com'))
                           .slice(0, 15)
            """)
            print(f"  [research] Google fallback returned {len(urls)} raw URL(s)", flush=True)
        except Exception as exc:
            print(f"  [research] Google fallback also failed: {exc}", flush=True)

    # ── Filter and cap ────────────────────────────────────────────────────
    clean = [u for u in urls if not _is_noise_url(u)]
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for u in clean:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    if len(deduped) < 3:
        print(
            f"  [research] ⚠ only {len(deduped)} seed URL(s) after noise filtering — continuing",
            flush=True,
        )

    return deduped[:n]


async def _depth2_links(page, base_url: str, topic: str, max_links: int = 3) -> list[str]:
    """
    Return up to max_links relevant internal links from the current page.

    Relevance filter: the link URL must contain at least one word from
    the topic query — this keeps depth-2 pages on-subject.
    """
    try:
        raw        = await get_page_links(page)
        base_host  = urlparse(base_url).netloc
        topic_words = {w for w in topic.lower().split() if len(w) > 3}

        links: list[str] = []
        for line in raw.splitlines():
            url = line.split(" | ")[0].strip()
            if not url.startswith("http"):
                continue
            if _is_noise_url(url):
                continue
            if urlparse(url).netloc != base_host:
                continue
            # Relevance: at least one meaningful topic word must appear in the URL
            if topic_words and not any(w in url.lower() for w in topic_words):
                continue
            links.append(url)
            if len(links) >= max_links:
                break
        return links
    except Exception:
        return []


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synthesize(query: str, corpus: str) -> str:
    """
    Call the local LLM to produce a research report from the gathered corpus.

    Uses the same import path as planner.py:
        from llm.main_llm import ask_llm

    The function is called via asyncio.to_thread so the blocking HTTP call
    to Ollama does not block the event loop.
    """
    try:
        from llm.main_llm import ask_llm   # same pattern as planner.py  # noqa: PLC0415

        system_prompt = (
            "You are a research synthesizer.  Write a comprehensive, "
            "well-structured report based on the provided web content.  "
            "Include key findings, facts, and a summary.  Be thorough.  "
            "Do NOT emit <start_plan> or <tool_call> tags.  "
            "Write in plain prose with clear section headings."
        )
        user_content = f"Topic: {query}\n\nSources:\n{corpus[:12_000]}"

        result = ask_llm(user_content, system_suffix=system_prompt)
        return result.strip() or "[Synthesis returned an empty response]"

    except Exception as exc:
        print(f"  [research] ⚠ synthesis failed: {exc}", flush=True)
        return f"[Synthesis failed — raw research below]\n\n{corpus[:4_000]}"


# ── Main research function ────────────────────────────────────────────────────

async def run_deep_research(
    query: str,
    save_path: Optional[str] = None,
    max_seeds: int = 5,
) -> str:
    """
    Full research pipeline.  Returns a markdown report string.
    If save_path is set the report is also written to that file.
    """
    manager = await BrowserManager.get()
    page    = await manager.page()

    print(f"  [research] starting deep research: {query!r}", flush=True)

    # Step 1: collect seed URLs from DuckDuckGo / Google
    seed_urls = await _seed_urls(page, query, n=max_seeds)
    if not seed_urls:
        return f"No search results found for: {query}"

    print(f"  [research] {len(seed_urls)} seed URL(s) after filtering", flush=True)

    corpus_parts: list[str] = []
    visited: set[str] = set()

    for seed_url in seed_urls:
        if seed_url in visited:
            continue
        visited.add(seed_url)

        # Depth-1: visit seed page
        print(f"  [research] visiting {seed_url}", flush=True)
        text = await _page_text(page, seed_url)
        if text and not text.startswith("[failed"):
            corpus_parts.append(f"[Source: {seed_url}]\n{text}")

        # Depth-2: follow relevant internal links
        depth2 = await _depth2_links(page, seed_url, query, max_links=3)
        for link in depth2:
            if link in visited:
                continue
            visited.add(link)
            print(f"  [research]   → depth-2 {link}", flush=True)
            text2 = await _page_text(page, link, max_chars=2_000)
            if text2 and not text2.startswith("[failed"):
                corpus_parts.append(f"[Source: {link}]\n{text2}")

    if not corpus_parts:
        return f"Could not retrieve any content for: {query}"

    corpus = "\n\n---\n\n".join(corpus_parts)
    print(
        f"  [research] synthesizing from {len(corpus_parts)} page(s) "
        f"({len(corpus):,} chars) …",
        flush=True,
    )

    # Step 3: synthesise with LLM (blocking call via thread so event loop stays free)
    report = await asyncio.to_thread(_synthesize, query, corpus)

    # Step 4: optionally save
    if save_path:
        try:
            path = Path(save_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
            report += f"\n\n---\n*Report saved to `{path}`*"
            print(f"  [research] saved to {path}", flush=True)
        except Exception as exc:
            report += f"\n\n[Warning: could not save report — {exc}]"

    return report


# ── Tool handler ──────────────────────────────────────────────────────────────

async def _deep_research(args: dict) -> str:
    query     = args.get("query", "")
    save_path = args.get("save_path")          # optional
    max_seeds = int(args.get("max_seeds", 5))

    if not query:
        return "Error: query is required."

    return await run_deep_research(query, save_path=save_path, max_seeds=max_seeds)


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name        = "deep_research",
    description = (
        "Perform multi-page web research on a topic: searches the web, visits the top "
        "result pages, follows relevant internal links one level deep, then synthesizes a "
        "structured report using AI.  Use this for thorough research tasks "
        "that require information from multiple web sources."
    ),
    parameters  = {
        "type": "object",
        "properties": {
            "query": {
                "type":        "string",
                "description": "The research question or topic to investigate",
            },
            "save_path": {
                "type":        "string",
                "description": "Optional file path to save the report (e.g. ~/Desktop/report.md)",
            },
            "max_seeds": {
                "type":        "integer",
                "description": "Maximum number of result pages to visit (default 5)",
                "default":     5,
            },
        },
        "required": ["query"],
    },
    handler = _deep_research,
))
