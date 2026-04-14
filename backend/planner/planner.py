"""backend/planner/planner.py — Fully autonomous multi-step task planner.

Architecture
────────────
The planner is a single async coroutine ``run_plan`` that runs as a background
asyncio Task in backend/main.py.  It operates in three phases:

  DECOMPOSE  — Ask qwen2.5:7b to break the goal into numbered steps.
  EXECUTE    — Run each step: ask the 7b model (with tool-call support) and
               optionally execute a tool from the ToolRegistry.
  SUMMARIZE  — Ask the 7b model to summarise what was accomplished, speak it
               aloud via TTS, and emit LLM_TOKEN events for the UI.

Two-model design
────────────────
  Conversational turns → qwen2.5:3b  (fast, handled by main_llm.py)
  All planner phases   → qwen2.5:7b  (higher reasoning capacity, avoids
                                       URL hallucination and goal drift)

The planner calls Ollama directly via _call_planner_llm() which uses
LLM_CONFIG.planner_model and LLM_CONFIG.planner_num_predict.  The normal
ask_llm / ask_llm_turn path is deliberately NOT used here — it would route
through the 3b conversational model.

All progress is broadcast to the frontend via PLAN_EVENT messages.

Concurrency model
─────────────────
run_plan() is an asyncio coroutine.  All blocking LLM calls are wrapped in
``asyncio.to_thread(...)`` so they don't block the event loop.  Tool
dispatch already runs natively async via ``registry.dispatch()``.

Cancellation
────────────
asyncio.CancelledError is caught at the top level; a PLAN_EVENT
{phase: "cancelled"} is broadcast and the coroutine returns cleanly.
"""

import asyncio
import json
import re
import threading
from typing import Callable

import requests

from config.llm import LLM_CONFIG, KEEP_ALIVE_IDLE, KEEP_ALIVE_PLAN
from llm.tool_executor import execute_step_with_tools as _execute_step_with_tools
from planner.validator import validate_steps as _validate_steps
from tools.registry import registry as _tool_registry
from utils.ram_monitor import can_load_7b, get_available_ram_gb


# ── Plan memory storage ───────────────────────────────────────────────────────

async def _store_plan_memory(
    goal: str,
    steps_done: list[str],
    summary: str,
    importance: float = 7.0,
) -> None:
    """
    Store what the planner accomplished as a PlannerMemory so future turns
    can answer "where did you save X?" and similar follow-up questions.

    importance=7.0 for completed plans (high — plans are significant events).
    importance=4.0 for partial/cancelled plans (lower — incomplete work).

    Runs in a background thread so it never delays the plan completion flow.
    Non-fatal — any exception is logged and swallowed.
    """
    from memory_system.core.insert_pipeline import insert_memory  # noqa: PLC0415

    content = (
        f"Completed task: {goal}\n"
        f"Steps taken: {'; '.join(steps_done)}\n"
        f"Result: {summary}"
    )

    def _run():
        try:
            insert_memory({
                "text":        content,
                "memory_type": "PlannerMemory",
                "source":      "planner",
            })
            print(f"  [planner] plan memory stored  ({len(content)} chars)", flush=True)
        except Exception as exc:
            print(f"  [planner] plan memory insert failed (non-fatal): {exc}", flush=True)

    await asyncio.to_thread(_run)

# ── Plan-active flag ──────────────────────────────────────────────────────────
# When True, Ollama calls use keep_alive="300s" to keep the model warm
# across all steps of the plan.  Resets to False when the plan finishes or
# is cancelled.  This avoids a 2-3s cold-load stall between every plan step.
_plan_active: bool = False

# ── Active plan model (set at run_plan() start based on RAM) ─────────────────
# Either LLM_CONFIG.planner_model (7b, preferred) or LLM_CONFIG.model (3b,
# fallback when RAM < 3 GB).  All per-plan LLM helpers read this variable.
_active_plan_model: str = ""


# ── Regex helpers ─────────────────────────────────────────────────────────────

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\)]\s+(.+)$", re.MULTILINE)

# Tag fragments that must never appear in a valid step description.
_BAD_STEP_TOKENS = ("<start_plan>", "</start_plan>", "<tool_call>", "</tool_call>")

# ── Multi-format tool call extractor for qwen2.5:7b ──────────────────────────
# The 7b model ignores <tool_call> tag instructions and uses various formats:
#   1. <tool_call>{"name":"x","args":{...}}</tool_call>  ← standard
#   2. {"name":"x","args":{...}}                         ← bare JSON
#   3. x({"key":"val"})                                  ← Python call style
#   4. x {"key":"val"}                                   ← name + JSON args

_TAGGED_TOOL_RE   = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BARE_JSON_RE     = re.compile(r'^\s*\{"name"\s*:\s*"([^"]+)"', re.DOTALL)
_PY_CALL_RE       = re.compile(r'^(\w+)\s*\((\{.*\})\)\s*$', re.DOTALL)
_NAME_JSON_RE     = re.compile(r'^(\w+)\s*(\{.*\})\s*$', re.DOTALL)


def _extract_7b_tool_call(
    text: str,
    known_names: set[str],
) -> tuple[str | None, dict | None]:
    """
    Multi-format tool call extractor for qwen2.5:7b.

    Attempts four parsing strategies in order of reliability.
    Returns (tool_name, tool_args) or (None, None) if no tool found.
    """
    stripped = text.strip()

    # Strategy 1: standard <tool_call>{...}</tool_call>
    m = _TAGGED_TOOL_RE.search(stripped)
    if m:
        try:
            payload = json.loads(m.group(1))
            name = payload.get("name")
            if name and (not known_names or name in known_names):
                return name, payload.get("args", {})
        except json.JSONDecodeError:
            pass

    # Strategy 2: bare JSON {"name": "...", "args": {...}}
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict) and "name" in payload:
                name = payload["name"]
                if not known_names or name in known_names:
                    return name, payload.get("args", {})
        except json.JSONDecodeError:
            pass

    # Strategy 3: Python-style tool_name({"key": "val"})
    m = _PY_CALL_RE.match(stripped)
    if m:
        try:
            name = m.group(1)
            args = json.loads(m.group(2))
            if isinstance(args, dict) and (not known_names or name in known_names):
                return name, args
        except json.JSONDecodeError:
            pass

    # Strategy 4: "tool_name {...}" — name followed by bare JSON args
    m = _NAME_JSON_RE.match(stripped)
    if m:
        try:
            name = m.group(1)
            args = json.loads(m.group(2))
            if isinstance(args, dict) and (not known_names or name in known_names):
                return name, args
        except json.JSONDecodeError:
            pass

    return None, None


# ── Direct Ollama call for the planner model ──────────────────────────────────
# Bypasses main_llm.py entirely so the planner always uses qwen2.5:7b
# regardless of what LLM_CONFIG.model is set to.

def _call_planner_llm_sync(
    system: str,
    user: str,
    max_tokens: int | None = None,
) -> str:
    """
    Blocking Ollama call using LLM_CONFIG.planner_model (qwen2.5:7b).

    Uses the planner's own num_predict budget (1024) by default, or a
    caller-supplied max_tokens for short yes/no checks (e.g. 5 tokens).

    keep_alive strategy:
      KEEP_ALIVE_PLAN while _plan_active=True  → model stays warm across all plan steps
      KEEP_ALIVE_IDLE when plan is done/cancelled → RAM freed within seconds
    """
    num_predict = max_tokens if max_tokens is not None else LLM_CONFIG.planner_num_predict
    keep_alive  = KEEP_ALIVE_PLAN if _plan_active else KEEP_ALIVE_IDLE
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    model = _active_plan_model or LLM_CONFIG.planner_model
    resp = requests.post(
        LLM_CONFIG.ollama_url,
        json={
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "keep_alive": keep_alive,
            "options":    {"num_predict": num_predict},
        },
        timeout=(LLM_CONFIG.stream_timeout_connect_s, LLM_CONFIG.stream_timeout_read_s),
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


async def _planner_llm(system: str, user: str, max_tokens: int | None = None) -> str:
    """Async wrapper around _call_planner_llm_sync — runs in a thread pool."""
    return await asyncio.to_thread(_call_planner_llm_sync, system, user, max_tokens)


# ── Direct Ollama streaming for executor (uses planner model) ─────────────────

def _stream_planner_ollama(messages: list[dict]):
    """Streaming Ollama call using the active plan model — yields token strings."""
    keep_alive = KEEP_ALIVE_PLAN if _plan_active else KEEP_ALIVE_IDLE
    model = _active_plan_model or LLM_CONFIG.planner_model
    resp = requests.post(
        LLM_CONFIG.ollama_url,
        json={
            "model":      model,
            "messages":   messages,
            "stream":     True,
            "keep_alive": keep_alive,
            "options":    {"num_predict": LLM_CONFIG.planner_num_predict},
        },
        stream=True,
        timeout=(LLM_CONFIG.stream_timeout_connect_s, LLM_CONFIG.stream_timeout_read_s),
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done", False):
                yield data.get("message", {}).get("content", "")


# ── Planner-specific system prompts ───────────────────────────────────────────

# Checker: used for yes/no binary questions — goal done? single-tool? on track?
_CHECKER_SYSTEM = (
    "Answer with exactly one word: YES or NO. "
    "Do not explain. Do not emit any tags. Just YES or NO."
)

# Summarizer: produces the spoken completion summary.
_SUMMARIZER_SYSTEM = (
    "You are a summarizer. Your only job is to write a short, clear summary "
    "of what was accomplished. "
    "NEVER emit <start_plan> tags. "
    "NEVER emit <tool_call> tags. "
    "NEVER ask questions. "
    "Write 2-3 sentences in first person, past tense, spoken directly to the user. "
    "Example: 'I researched local LLMs and saved a report to your desktop. "
    "The report covers the latest open-source models and their benchmarks.'"
)

# Executor: injected as the system role for every per-step LLM call.
# Must be completely separate from the orchestrator SYSTEM_PROMPT so
# <start_plan> trigger instructions never reach the executor.
_EXECUTOR_SYSTEM = """\
You are executing a single step in a plan.

OUTPUT RULES — read carefully:
- If you need a tool: output ONLY the tool call block, nothing else
- If no tool needed: output ONLY the plain text answer, nothing else
- NEVER output explanations before or after a tool call
- NEVER output these words: "WRONG formats", "never use these", \
"tool_call syntax", "example", "rules", "instructions"
- NEVER emit <start_plan> tags
- NEVER emit multiple tool calls in one response
- NEVER hallucinate tool results — if a tool fails, say "failed: reason"
"""

# Verbatim decomposition prompt from spec — do not paraphrase.
_DECOMPOSE_SYSTEM = "You are a task decomposer. Respond ONLY with a numbered list."

_DECOMPOSE_PROMPT_TEMPLATE = """\
You are a task decomposer. Your only job is to \
break a goal into a numbered list of concrete, executable steps.

RULES:
- Each step must be a single, specific action
- Each step must directly relate to the goal
- Each step must be at least 6 words — be specific about what you are doing
- Use only these action types: navigate to URL, read page content to find X, \
search for EXACT QUERY, click element SELECTOR, type text VALUE, \
write file PATH with content, read file PATH, run command CMD
- Maximum 6 steps
- No explanation, no preamble, no commentary
- No code, no tool syntax, no JSON — plain English steps only
- If a step would use a web search, write the EXACT query
- If a step would navigate, write the EXACT URL (no made-up domains)
- Never hallucinate URLs — only use real, well-known domains
- After navigating to a page, ALWAYS include a "read page content to find X" step next
- NEVER use web search when you could navigate directly to the page

GOAL: {goal}

Respond with ONLY the numbered list. Nothing else."""


# ── Step parsing ──────────────────────────────────────────────────────────────

def _parse_steps(text: str) -> list[str]:
    """Extract and coarse-filter numbered step lines from decomposer output."""
    matches = _NUMBERED_LINE_RE.findall(text)
    steps: list[str] = []
    for m in matches:
        step = m.strip()
        if not step:
            continue
        if len(step) < 10:
            print(f"  [planner] ⚠ filtered bad step (too short): {step!r}", flush=True)
            continue
        if any(tok in step for tok in _BAD_STEP_TOKENS):
            print(f"  [planner] ⚠ filtered bad step (contains tag): {step!r}", flush=True)
            continue
        steps.append(step)
    return steps


# ── Goal-completion check ─────────────────────────────────────────────────────

async def _check_goal_done(goal: str, results: list[str]) -> bool:
    """Ask the 7b model: is the goal fully achieved? Returns True on YES."""
    summary = "\n".join(f"- {r[:200]}" for r in results[-5:])
    prompt = (
        f"Goal: {goal}\n\n"
        f"Results so far:\n{summary}\n\n"
        "Is the goal fully achieved? Answer YES or NO and nothing else."
    )
    # max_tokens=5 — only need YES or NO, saves latency
    answer = await _planner_llm(_CHECKER_SYSTEM, prompt, max_tokens=5)
    return "yes" in answer.strip().lower()[:10]


# ── Drift detection ───────────────────────────────────────────────────────────

async def _check_on_track(goal: str, recent_results: list[str]) -> bool:
    """
    Ask the 7b model: are the recent results still working toward the goal?
    Returns True if on track, False if drifted.
    Called every 2 steps to catch hallucination spirals early.
    """
    prompt = (
        f"Goal: {goal}\n\n"
        f"Recent actions: {' | '.join(r[:150] for r in recent_results[-2:])}\n\n"
        "Are these actions making progress toward the goal?\n"
        "Answer YES or NO only."
    )
    response = await _planner_llm(_CHECKER_SYSTEM, prompt, max_tokens=5)
    return "yes" in response.lower()


# ── Per-step executor (uses planner model) ────────────────────────────────────

async def _execute_step_planner(
    step: str,
    context: str,
    correction: str = "",
) -> str:
    """
    Execute one step using qwen2.5:7b.

    Mirrors tool_executor.execute_step_with_tools() but routes through
    _stream_planner_ollama() so the 7b model is used instead of 3b.
    The correction parameter injects a drift-recovery notice when needed.
    """
    from tools.registry import registry as _registry  # noqa: PLC0415

    # Build ordered tool schema: web tools first to bias LLM preference.
    all_schemas      = _registry.get_schemas()
    web_schemas      = [s for s in all_schemas if s["name"].startswith("web_")]
    research_schemas = [s for s in all_schemas if s["name"] == "deep_research"]
    other_schemas    = [s for s in all_schemas
                        if not s["name"].startswith("web_") and s["name"] != "deep_research"]
    schemas_json = json.dumps(web_schemas + research_schemas + other_schemas, indent=2)

    # Limit context to last 2 results, max 300 chars total to prevent overflow.
    ctx_parts = context.split("\n")[-2:] if context else []
    ctx_short = " | ".join(ctx_parts)
    if len(ctx_short) > 300:
        ctx_short = ctx_short[-300:]

    # Inject correction notice at the top when drift was detected.
    correction_block = f"\n{correction}\n" if correction else ""

    user_prompt = "\n".join(filter(None, [
        correction_block or "",
        "## TOOL CALL FORMAT — use this EXACTLY:",
        "<tool_call>",
        '{"name": "TOOL_NAME", "args": {"param": "value"}}',
        "</tool_call>",
        "",
        "You MUST wrap tool calls in <tool_call> tags as shown above.",
        "Do NOT output bare JSON. Do NOT use Python function call syntax.",
        "If no tool is needed, output plain text only.",
        "",
        "## Web browsing rules",
        "ALWAYS use web_navigate / web_read / web_search for web tasks.",
        "NEVER use fetch_url for real websites.",
        "",
        "## Available tools (web tools listed first):",
        schemas_json,
        "",
        f"## Step to execute: {step}",
        f"## Context from previous steps: {ctx_short or 'None yet'}",
    ]))

    # Pass 1: collect full response from 7b model.
    def _pass1():
        msgs = [
            {"role": "system", "content": _EXECUTOR_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]
        tokens: list[str] = []
        for tok in _stream_planner_ollama(msgs):
            tokens.append(tok)
        full = "".join(tokens)
        return full, tokens

    full_text, _tokens = await asyncio.to_thread(_pass1)
    print(
        f"  [executor/7b] response ({len(full_text)} chars): {full_text[:100]!r}",
        flush=True,
    )

    # Use the multi-format extractor — 7b uses 4 different output styles.
    known = {s["name"] for s in _registry.get_schemas()}
    tool_name, tool_args = _extract_7b_tool_call(full_text, known)
    if tool_name:
        print(f"  [executor/7b] ✓ tool call parsed: {tool_name}", flush=True)

    if tool_name is None:
        # No tool call — plain-text answer.
        return full_text.strip()

    # Tool detected — dispatch.
    print(f"  [executor] 🔧 tool: {tool_name}  args={tool_args}", flush=True)
    tool_result = await _registry.dispatch(tool_name, tool_args or {})
    print(f"  [executor] 🔧 result ({len(tool_result)} chars): {tool_result[:120]}", flush=True)

    # Pass 2: one-sentence summary of the tool result using 7b model.
    def _pass2():
        msgs2 = [
            {"role": "system",    "content": _EXECUTOR_SYSTEM},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": full_text},
            {"role": "tool",      "content": f"Tool '{tool_name}' result:\n{tool_result}"},
            {
                "role":    "user",
                "content": (
                    "Summarise the tool result in one sentence as the step outcome. "
                    "Plain text only — no tags, no narration."
                ),
            },
        ]
        tokens2: list[str] = []
        for tok in _stream_planner_ollama(msgs2):
            tokens2.append(tok)
        return "".join(tokens2).strip()

    return await asyncio.to_thread(_pass2)


# ── Main planner coroutine ────────────────────────────────────────────────────

async def run_plan(
    goal: str,
    broadcast: Callable,
    max_steps: int = 20,
) -> None:
    """
    Execute a multi-step autonomous plan for the given goal.

    Parameters
    ----------
    goal:       One-sentence description of what to accomplish.
    broadcast:  Callable matching _emit(event: str, data: dict) from main.py.
    max_steps:  Hard cap on execution steps to prevent infinite loops.
    """
    global _plan_active, _active_plan_model

    # ── RAM-aware model selection ─────────────────────────────────────────────
    available_gb = get_available_ram_gb()
    if can_load_7b():
        _active_plan_model = LLM_CONFIG.planner_model
        print(
            f"\n  [planner] 🗺 starting plan  model={_active_plan_model}"
            f"  RAM={available_gb:.1f}GB free: '{goal}'",
            flush=True,
        )
    else:
        _active_plan_model = LLM_CONFIG.model   # fall back to 3b
        print(
            f"\n  [planner] ⚠ low RAM ({available_gb:.1f}GB) — falling back to "
            f"{_active_plan_model}: '{goal}'",
            flush=True,
        )
        broadcast("SYSTEM_EVENT", {
            "event":        "low_memory",
            "message":      f"Low RAM ({available_gb:.1f} GB free) — using fast model, plan quality may be reduced",
            "available_gb": round(available_gb, 2),
        })

    _plan_active = True   # keep model warm across all plan steps (keep_alive=300s)

    try:
        # ── Pre-plan: guard against trivially single-step goals ──────────
        # Only cancels if the goal is clearly one-shot (e.g. a reminder,
        # a single write, a single search).  Multi-step web tasks that
        # involve navigation + reading always need the planner.
        _check_prompt = (
            f"Does completing this goal require 2 or more separate tool actions?\n"
            f"Examples of tool actions: web_navigate, web_read, web_search, "
            f"write_file, run_terminal, set_reminder.\n"
            f"Note: navigating to a page AND then reading it = 2 tool actions.\n"
            f"Note: searching AND then reading a result = 2 tool actions.\n"
            f"Goal: {goal}\n"
            "Answer YES or NO only."
        )
        _check_ans = await _planner_llm(_CHECKER_SYSTEM, _check_prompt, max_tokens=5)
        # Only bypass the planner if the answer is an unambiguous NO
        if _check_ans.strip().lower().startswith("no"):
            print("  [plan] cancelled — single-tool goal, routing to normal turn", flush=True)
            # Execute the goal as a single step and stream the result.
            result = await _execute_step_planner(goal, "")
            broadcast("VOICE_STATE", {"state": "speaking"})
            words = result.split()
            for j, word in enumerate(words):
                broadcast("LLM_TOKEN", {"token": word if j == 0 else " " + word, "done": False})
                await asyncio.sleep(0.005)
            broadcast("LLM_TOKEN", {"token": "", "done": True, "token_count": len(words)})
            try:
                from tts import speak as _speak  # noqa: PLC0415
                await asyncio.to_thread(_speak, result, threading.Event())
            except Exception as exc:
                print(f"  [planner] ⚠ TTS speak failed (single-step): {exc}", flush=True)
            broadcast("VOICE_STATE", {"state": "idle"})
            return

        # ── Phase 1: DECOMPOSE using 7b model ─────────────────────────────
        decompose_user = _DECOMPOSE_PROMPT_TEMPLATE.format(goal=goal)
        decompose_resp = await _planner_llm(_DECOMPOSE_SYSTEM, decompose_user)

        steps = _parse_steps(decompose_resp)
        # Coarse tag/length filter already applied in _parse_steps.
        # Now run the richer semantic validator from validator.py.
        steps = _validate_steps(steps, goal)

        if not steps:
            # Neither parse nor validate found usable steps — treat goal as one step.
            steps = [goal]

        steps = steps[:max_steps]

        print(f"  [planner] decomposed into {len(steps)} step(s)", flush=True)
        broadcast("PLAN_EVENT", {
            "phase": "decomposed",
            "steps": steps,
            "goal":  goal,
        })

        # ── Phase 2: EXECUTE ───────────────────────────────────────────────
        results:        list[str] = []
        context_window: list[str] = []   # rolling last-3 results as context
        consecutive_drifts = 0           # track back-to-back drift signals

        for i, step in enumerate(steps):
            print(f"  [planner] step {i+1}/{len(steps)}: {step[:80]}", flush=True)
            broadcast("PLAN_EVENT", {
                "phase":      "step_start",
                "step_index": i,
                "step_text":  step,
                "total":      len(steps),
            })

            # Build correction notice if previous drift check fired.
            correction = ""
            if consecutive_drifts > 0:
                next_step = steps[i] if i < len(steps) else ""
                correction = (
                    f"CORRECTION: The previous steps drifted from the goal. "
                    f"Refocus. Goal: {goal}. Execute only what's needed for: {next_step}"
                )

            try:
                context = "\n".join(context_window[-3:])
                # Route through the 7b executor instead of the 3b tool_executor.
                result  = await _execute_step_planner(step, context, correction=correction)
            except Exception as exc:
                result = f"Error on step {i+1}: {exc}"
                print(f"  [planner] ⚠ step error: {exc}", flush=True)

            results.append(result)
            context_window.append(f"Step {i+1} ({step[:50]}): {result[:200]}")

            print(f"  [planner] step {i+1} done: {result[:80]}", flush=True)
            broadcast("PLAN_EVENT", {
                "phase":      "step_done",
                "step_index": i,
                "result":     result,
            })

            # ── Drift detection every 2 steps ──────────────────────────────
            # After step 2, 4, 6, … ask the 7b model if we're still on track.
            if i % 2 == 0 and i > 0:
                try:
                    on_track = await _check_on_track(goal, results)
                    if not on_track:
                        consecutive_drifts += 1
                        print(
                            f"  [planner] drift detected at step {i+1} "
                            f"(consecutive={consecutive_drifts})",
                            flush=True,
                        )
                        if consecutive_drifts >= 2:
                            # Two consecutive drift signals — abort.
                            raise RuntimeError(
                                "lost track of goal after correction attempt"
                            )
                        # Single drift: inject correction into next step (handled above).
                    else:
                        consecutive_drifts = 0   # reset on clean check
                except RuntimeError:
                    raise   # re-raise drift abort
                except Exception:
                    pass    # drift check failure is non-fatal

            # ── Early exit: goal already achieved ──────────────────────────
            if len(results) >= 2:
                try:
                    done = await _check_goal_done(goal, results)
                    if done:
                        print(f"  [planner] ✓ goal achieved after {i+1} step(s)", flush=True)
                        break
                except Exception:
                    pass   # goal-check failure is non-fatal

        # ── Phase 3: SUMMARIZE using 7b model ─────────────────────────────
        all_results = "\n".join(f"Step {j+1}: {r}" for j, r in enumerate(results))
        summary_user = (
            "Summarise what was accomplished for this goal in 2-3 sentences, "
            "speaking directly to the user in first person as their assistant. "
            "Be concrete about what was done. Voice output only — no lists or markdown.\n\n"
            f"Goal: {goal}\n\nResults:\n{all_results}"
        )
        summary = await _planner_llm(_SUMMARIZER_SYSTEM, summary_user)
        summary = summary.strip()
        # Safety net: strip any control tags that escaped the system-prompt guard.
        summary = re.sub(r"<start_plan>.*?</start_plan>", "", summary, flags=re.DOTALL).strip()
        summary = re.sub(r"<tool_call>.*?</tool_call>",   "", summary, flags=re.DOTALL).strip()

        print(f"  [planner] ✅ complete. Summary: {summary[:100]}", flush=True)
        broadcast("PLAN_EVENT", {
            "phase":   "complete",
            "summary": summary,
            "goal":    goal,
        })

        # ── Store plan memory (background) ────────────────────────────────
        # Captures goal + steps + summary so future turns can recall what
        # the planner did (e.g. "where did you save that file?").
        asyncio.ensure_future(
            _store_plan_memory(goal, results, summary, importance=7.0)
        )

        # ── Add to short-term context buffer ──────────────────────────────
        # Immediately visible in the NEXT turn's LLM prompt via recent_actions.
        try:
            from main import add_recent_action  # noqa: PLC0415
            add_recent_action(f"Completed: {goal} → {summary[:200]}")
        except Exception:
            pass   # main.py import may not be available in test contexts

        # Stream summary as word-level LLM_TOKEN events for the UI.
        broadcast("VOICE_STATE", {"state": "speaking"})
        words = summary.split()
        for j, word in enumerate(words):
            token = word if j == 0 else " " + word
            broadcast("LLM_TOKEN", {"token": token, "done": False})
            await asyncio.sleep(0.01)
        broadcast("LLM_TOKEN", {"token": "", "done": True, "token_count": len(words)})

        # Speak the summary aloud via TTS.
        try:
            from tts import speak as _speak  # noqa: PLC0415
            await asyncio.to_thread(_speak, summary, threading.Event())
        except Exception as exc:
            print(f"  [planner] ⚠ TTS speak failed: {exc}", flush=True)

        broadcast("VOICE_STATE", {"state": "idle"})

    except asyncio.CancelledError:
        _plan_active = False   # evict 7b model immediately
        print("  [planner] ⛔ cancelled", flush=True)
        broadcast("PLAN_EVENT", {"phase": "cancelled", "goal": goal})
        broadcast("VOICE_STATE", {"state": "idle"})
        # Store partial completion if any steps finished — importance=4.0 (partial)
        if results:
            partial_summary = f"Cancelled after {len(results)} step(s)."
            asyncio.ensure_future(
                _store_plan_memory(goal, results, partial_summary, importance=4.0)
            )
            try:
                from main import add_recent_action  # noqa: PLC0415
                add_recent_action(f"Cancelled (partial): {goal} — {len(results)} steps done")
            except Exception:
                pass
        raise

    except Exception as exc:
        _plan_active = False   # evict 7b model immediately
        import traceback
        reason = str(exc)
        print(f"  [planner] ✗ failed: {reason}", flush=True)
        traceback.print_exc()
        broadcast("PLAN_EVENT", {
            "phase":  "failed",
            "reason": reason,
            "goal":   goal,
        })
        broadcast("LLM_TOKEN", {
            "token": f"I hit an error while working on that: {reason}", "done": False,
        })
        broadcast("LLM_TOKEN", {"token": "", "done": True})
        broadcast("VOICE_STATE", {"state": "idle"})

    finally:
        _plan_active = False   # always reset — even if plan completes normally
