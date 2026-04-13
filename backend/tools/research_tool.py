"""backend/tools/research_tool.py — Deep web research tool.

Performs multi-page research:
  1. Google search for the query via web_agent search_google action
  2. Visit the top N result pages and extract their text
  3. Follow one level of internal links per page (depth-2 crawl)
  4. Synthesize everything with a local LLM call
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
from pathlib import Path
from typing import Optional

from web_agent.browser import BrowserManager
from web_agent.actions import search_google, navigate, get_page_text, get_page_links
from tools.registry import registry, ToolDefinition

# LLM used for synthesis — import lazily to avoid circular imports
_llm_call = None   # set by _get_llm_call() on first use


def _get_llm_call():
    global _llm_call
    if _llm_call is None:
        from llm.ollama_client import stream_ollama  # noqa: PLC0415
        _llm_call = stream_ollama
    return _llm_call


# ── Crawl helpers ─────────────────────────────────────────────────────────────

async def _page_text(page, url: str, max_chars: int = 4_000) -> str:
    """Navigate to url, return capped page text."""
    result = await navigate(page, url)
    if "failed" in result.lower():
        return f"[failed to load {url}]"
    text = await get_page_text(page, max_chars)
    return text


async def _seed_urls(page, query: str, n: int = 5) -> list[str]:
    """Run a Google search and return up to n result URLs."""
    raw = await search_google(page, query)
    urls: list[str] = []
    for line in raw.splitlines():
        # format: "Title — URL"
        if " — " in line:
            url = line.split(" — ")[-1].strip()
            if url.startswith("http") and "google.com" not in url:
                urls.append(url)
        if len(urls) >= n:
            break
    return urls


async def _depth2_links(page, base_url: str, max_links: int = 3) -> list[str]:
    """Return up to max_links internal links from the current page."""
    try:
        raw = await get_page_links(page)
        from urllib.parse import urlparse
        base_host = urlparse(base_url).netloc
        links: list[str] = []
        for line in raw.splitlines():
            url = line.split(" | ")[0].strip()
            if url.startswith("http") and urlparse(url).netloc == base_host:
                links.append(url)
            if len(links) >= max_links:
                break
        return links
    except Exception:
        return []


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synthesize(query: str, corpus: str) -> str:
    """Call the local LLM to produce a research report from gathered text."""
    prompt = (
        f"You are a research assistant.  Based only on the web content below, "
        f"write a thorough, well-structured markdown report answering this research "
        f"query: \"{query}\"\n\n"
        f"Include key findings, relevant facts, and cite sources where possible.\n\n"
        f"=== WEB CONTENT ===\n{corpus[:12_000]}\n=== END CONTENT ===\n\n"
        f"Report:"
    )
    try:
        stream_ollama = _get_llm_call()
        tokens = list(stream_ollama(
            system_prompt="You are a helpful research assistant who writes clear markdown reports.",
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        ))
        return "".join(tokens)
    except Exception as exc:
        return f"Synthesis failed: {exc}\n\n[Raw corpus below]\n{corpus[:4000]}"


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

    # Step 1: collect seed URLs from Google
    seed_urls = await _seed_urls(page, query, n=max_seeds)
    if not seed_urls:
        return f"No search results found for: {query}"

    print(f"  [research] {len(seed_urls)} seed URL(s)", flush=True)

    corpus_parts: list[str] = []
    visited: set[str] = set()

    for seed_url in seed_urls:
        if seed_url in visited:
            continue
        visited.add(seed_url)

        # Depth-1: visit seed page
        print(f"  [research] visiting {seed_url}", flush=True)
        text = await _page_text(page, seed_url)
        if text:
            corpus_parts.append(f"[Source: {seed_url}]\n{text}")

        # Depth-2: follow internal links
        depth2 = await _depth2_links(page, seed_url, max_links=3)
        for link in depth2:
            if link in visited:
                continue
            visited.add(link)
            print(f"  [research]   → depth-2 {link}", flush=True)
            text2 = await _page_text(page, link, max_chars=2_000)
            if text2:
                corpus_parts.append(f"[Source: {link}]\n{text2}")

    if not corpus_parts:
        return f"Could not retrieve any content for: {query}"

    corpus = "\n\n---\n\n".join(corpus_parts)
    print(f"  [research] synthesizing from {len(corpus_parts)} page(s) …", flush=True)

    # Step 3: synthesise with LLM (blocking call via thread)
    report = await asyncio.to_thread(_synthesize, query, corpus)

    # Step 4: optionally save
    if save_path:
        try:
            path = Path(save_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report)
            report += f"\n\n---\n*Report saved to `{path}`*"
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
        "Perform multi-page web research on a topic: searches Google, visits the top "
        "result pages, follows internal links one level deep, then synthesizes a "
        "structured markdown report using AI.  Use this for thorough research tasks "
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
                "description": "Optional file path to save the report (e.g. ~/research/topic.md)",
            },
            "max_seeds": {
                "type":        "integer",
                "description": "Maximum number of Google result pages to visit (default 5, max 5)",
                "default":     5,
            },
        },
        "required": ["query"],
    },
    handler = _deep_research,
))
