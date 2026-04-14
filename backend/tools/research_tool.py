"""backend/tools/research_tool.py — Deep web research tool.

Uses the Playwright-backed web_* tools via the ToolRegistry so that:
  • All browsing is visible in the browser window
  • JS-heavy pages render correctly
  • Results are consistent with what the user sees

Pipeline
────────
  1. web_search  → get seed URLs (Playwright, real browser)
  2. web_navigate + web_read → visit each seed page
  3. web_links + depth-2 relevance filter → follow sub-links
  4. Direct async Ollama call → synthesize report
     (bypasses ask_llm_turn to avoid plan-trigger injection)
  5. Optionally write_file to save the report

Bug-3b guard: the synthesis call uses a stripped prompt with no
orchestrator behavior, so the LLM cannot emit <tool_call> or
<start_plan> tags in the report.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Optional

from config.llm import LLM_CONFIG
from tools.registry import registry, ToolDefinition

import httpx


# ── URL noise filter ──────────────────────────────────────────────────────────

_NOISE_DOMAINS = (
    "youtube.com", "youtu.be",
    "google.com/products", "google.com/intl",
    "accounts.google.com", "play.google.com", "support.google.com",
    "facebook.com", "instagram.com",
    "twitter.com", "x.com",
    "tiktok.com", "reddit.com/r/",
)


def _is_noise(url: str) -> bool:
    lower = url.lower()
    return any(pat in lower for pat in _NOISE_DOMAINS)


def _parse_search_urls(raw: str) -> list[str]:
    """Parse URLs from web_search result ('Title — URL\\n...' format)."""
    urls: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        url = ""
        if " — http" in line:
            url = line.split(" — ", 1)[1].strip()
        elif line.strip().startswith("http"):
            url = line.strip()
        if url and url not in seen and not _is_noise(url):
            seen.add(url)
            urls.append(url)
    return urls


# ── Direct async Ollama synthesis ─────────────────────────────────────────────

async def _synthesize_async(topic: str, all_text: str) -> str:
    """
    Call Ollama directly (async, non-streaming) with a stripped system prompt.

    Does NOT go through ask_llm_turn so there is zero chance of the LLM
    emitting <tool_call> or <start_plan> tags inside the research report.
    """
    if len(all_text) > 24_000:
        all_text = all_text[:24_000] + "\n[truncated for length]"

    synthesis_prompt = (
        f"You are a research synthesizer. Write a comprehensive, "
        f"well-structured report on: {topic}\n\n"
        f"Base it entirely on these sources:\n\n{all_text}\n\n"
        f"Structure your report with: Executive Summary, Key Findings, "
        f"Detailed Analysis, Sources.\n"
        f"Be thorough and factual. Do NOT emit XML tags, tool calls, "
        f"or plan triggers. Write the report text directly."
    )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                LLM_CONFIG.ollama_url,
                json={
                    "model":      LLM_CONFIG.model,
                    "messages":   [{"role": "user", "content": synthesis_prompt}],
                    "stream":     False,
                    "keep_alive": 0,   # evict after synthesis — frees RAM immediately
                    "options":    {"num_predict": 1500},
                },
            )
            resp.raise_for_status()
            data    = resp.json()
            report  = data["message"]["content"]
    except Exception as exc:
        print(f"  [research] ⚠ synthesis failed: {exc}", flush=True)
        return f"[Synthesis failed — raw research below]\n\n{all_text[:4_000]}"

    # Strip any accidentally emitted control tags
    report = re.sub(r"<start_plan>.*?</start_plan>", "", report, flags=re.DOTALL)
    report = re.sub(r"<tool_call>.*?</tool_call>",   "", report, flags=re.DOTALL)
    return report.strip()


# ── Main research coroutine ───────────────────────────────────────────────────

async def run_deep_research(
    topic: str,
    depth: int = 2,
    max_pages: int = 8,
    save_to: Optional[str] = None,
) -> str:
    """
    Full research pipeline using Playwright web_* tools.

    All browsing goes through registry.dispatch() so the real Chromium
    window is used — pages render with JS and are visible to the user.
    """
    depth     = min(depth, 3)
    max_pages = min(max_pages, 15)

    print(f"  [research] starting: {topic!r}  depth={depth}  max_pages={max_pages}", flush=True)

    collected: list[dict] = []   # {"url": str, "text": str}

    # ── Step 1: search ────────────────────────────────────────────────────
    print("  [research] step 1: web_search via Playwright", flush=True)
    try:
        search_raw = await registry.dispatch("web_search", {"query": topic})
        seed_urls  = _parse_search_urls(search_raw)
    except Exception as exc:
        return f"Research failed during search: {exc}"

    if not seed_urls:
        return f"No usable search results found for: {topic}"

    seed_urls = seed_urls[:max_pages]
    print(f"  [research] {len(seed_urls)} seed URL(s) after noise filter", flush=True)

    # ── Step 2: visit each seed page ─────────────────────────────────────
    topic_words = {w for w in topic.lower().split() if len(w) > 3}

    for url in seed_urls:
        if len(collected) >= max_pages:
            break
        try:
            print(f"  [research] navigate → {url}", flush=True)
            await registry.dispatch("web_navigate", {"url": url})
            text = await registry.dispatch("web_read", {"max_chars": 6_000})
            if text and not text.startswith("Error"):
                collected.append({"url": url, "text": text})
            else:
                print(f"  [research]   ⚠ empty/error read: {text[:60]}", flush=True)
        except Exception as exc:
            print(f"  [research]   ⚠ failed to visit {url}: {exc}", flush=True)
            collected.append({"url": url, "text": f"Failed to load: {exc}"})
            continue

        # ── Depth-2: follow relevant sub-links ────────────────────────────
        if depth >= 2 and len(collected) < max_pages:
            try:
                links_raw = await registry.dispatch("web_links", {})
                sub_candidates: list[str] = []
                for line in links_raw.splitlines()[:80]:
                    sub_url = line.split(" | ")[0].strip() if " | " in line else line.strip()
                    if not sub_url.startswith("http"):
                        continue
                    if _is_noise(sub_url):
                        continue
                    # Relevance: URL must contain at least one topic word
                    if topic_words and not any(w in sub_url.lower() for w in topic_words):
                        continue
                    sub_candidates.append(sub_url)

                for sub_url in sub_candidates[:3]:
                    if len(collected) >= max_pages:
                        break
                    try:
                        print(f"  [research]   → depth-2 {sub_url}", flush=True)
                        await registry.dispatch("web_navigate", {"url": sub_url})
                        sub_text = await registry.dispatch("web_read", {"max_chars": 4_000})
                        if sub_text and not sub_text.startswith("Error"):
                            collected.append({"url": sub_url, "text": sub_text})
                    except Exception:
                        continue
            except Exception as exc:
                print(f"  [research]   ⚠ depth-2 link extraction failed: {exc}", flush=True)

    if not collected:
        return f"Could not retrieve any page content for: {topic}"

    print(
        f"  [research] collected {len(collected)} page(s) — synthesizing …",
        flush=True,
    )

    # ── Step 3: synthesize ────────────────────────────────────────────────
    all_text = "\n\n---\n\n".join(
        f"Source: {c['url']}\n{c['text']}" for c in collected
    )
    report = await _synthesize_async(topic, all_text)

    # ── Step 4: optionally save ───────────────────────────────────────────
    if save_to:
        try:
            save_result = await registry.dispatch(
                "write_file",
                {"path": save_to, "content": report, "mode": "overwrite"},
            )
            print(f"  [research] saved: {save_result}", flush=True)
            summary_preview = report[:500].replace("\n", " ")
            return (
                f"Research complete. Report saved to {save_to}\n\n"
                f"Summary: {summary_preview}…"
            )
        except Exception as exc:
            report += f"\n\n[Warning: could not save — {exc}]"

    return report


# ── Tool handler ──────────────────────────────────────────────────────────────

async def _deep_research(args: dict) -> str:
    # Accept both "query" (old) and "topic" (new canonical name)
    topic     = args.get("topic") or args.get("query", "")
    depth     = int(args.get("depth", 2))
    max_pages = int(args.get("max_pages", 8))
    save_to   = args.get("save_to") or args.get("save_path")   # support both keys

    if not topic:
        return "Error: topic (or query) is required."

    return await run_deep_research(
        topic     = topic,
        depth     = depth,
        max_pages = max_pages,
        save_to   = save_to,
    )


# ── Self-registration ─────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name        = "deep_research",
    description = (
        "Perform multi-page web research on a topic using a real browser: searches, "
        "visits result pages, follows relevant sub-links, then synthesizes a structured "
        "AI report. Use for thorough research requiring multiple web sources. "
        "Optionally saves the report to a file."
    ),
    parameters  = {
        "type": "object",
        "properties": {
            "topic": {
                "type":        "string",
                "description": "The research topic or question to investigate",
            },
            "depth": {
                "type":        "integer",
                "description": "Link follow depth (1=seed pages only, 2=follow sub-links). Default 2.",
                "default":     2,
            },
            "max_pages": {
                "type":        "integer",
                "description": "Maximum total pages to visit (default 8, max 15)",
                "default":     8,
            },
            "save_to": {
                "type":        "string",
                "description": "Optional file path to save the report (e.g. ~/Desktop/report.md)",
            },
        },
        "required": ["topic"],
    },
    handler = _deep_research,
))
