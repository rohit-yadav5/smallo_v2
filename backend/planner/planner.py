"""backend/planner/planner.py — Fully autonomous multi-step task planner.

Architecture
────────────
The planner is a single async coroutine ``run_plan`` that runs as a background
asyncio Task in backend/main.py.  It operates in three phases:

  DECOMPOSE  — Ask the LLM to break the goal into numbered steps.
  EXECUTE    — Run each step: ask the LLM (with tool-call support) and
               optionally execute a tool from the ToolRegistry.
  SUMMARIZE  — Ask the LLM to summarise what was accomplished, speak it
               aloud via TTS, and emit LLM_TOKEN events for the UI.

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

Phase 3 note
────────────
In Phase 3, specialist sub-agents (web research, DevOps, code review) will
be invoked here instead of the general-purpose ToolRegistry.  The step
execution loop is the natural injection point — replace ``_execute_step``
with a sub-agent dispatcher.
"""

import asyncio
import re
import threading
from typing import Callable

# LLM calls are synchronous (requests-based); we wrap them in asyncio.to_thread.
from llm.main_llm import ask_llm

# Shared tool-detection + execution logic (avoids duplicating the two-pass
# tool-call flow that also lives in main_llm.py / ask_llm_turn).
from llm.tool_executor import execute_step_with_tools as _execute_step_with_tools

# Tool registry — queried at run time (after tools are registered by main.py)
# to give the decomposer an accurate list of available tool names.
from tools.registry import registry as _tool_registry


# ── Regex helpers ─────────────────────────────────────────────────────────────

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\)]\s+(.+)$", re.MULTILINE)

# Tag fragments that must never appear in a valid step description.
# The LLM sometimes copies planner-trigger or tool-call syntax into step text.
_BAD_STEP_TOKENS = ("<start_plan>", "</start_plan>", "<tool_call>", "</tool_call>")


def _parse_and_filter_steps(text: str) -> list[str]:
    """
    Extract numbered step lines from an LLM decomposition response and
    filter out malformed entries.

    Filtered (with warning log) if a step:
      - is fewer than 10 characters (too vague to be actionable)
      - contains a planner-trigger or tool-call tag (the LLM accidentally
        copied format syntax into the step text)
    """
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


# ── Stripped system prompts for planner-internal LLM calls ───────────────────
# These calls must NOT use the orchestrator system prompt, which contains
# <start_plan> trigger instructions.  Using a minimal system prompt here
# prevents the LLM from emitting plan triggers inside an already-running plan.

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

_CHECKER_SYSTEM = (
    "Answer with exactly one word: YES or NO. "
    "Do not explain. Do not emit any tags. Just YES or NO."
)


# ── LLM helper (runs via asyncio.to_thread) ───────────────────────────────────

async def _llm(prompt: str, system: str = "") -> str:
    """One-shot async LLM call; returns full response text."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    return await asyncio.to_thread(ask_llm, full_prompt)


async def _check_goal_done(goal: str, results: list[str]) -> bool:
    """Ask the LLM: is the goal fully achieved?  Returns True on YES."""
    summary = "\n".join(f"- {r}" for r in results[-5:])   # last 5 results
    prompt = (
        f"Goal: {goal}\n\n"
        f"Results so far:\n{summary}\n\n"
        "Is the goal fully achieved?  Answer YES or NO and nothing else."
    )
    # Use the stripped checker system prompt to prevent plan trigger emission
    full_prompt = f"{_CHECKER_SYSTEM}\n\n{prompt}"
    answer = await asyncio.to_thread(ask_llm, full_prompt)
    return "yes" in answer.strip().lower()[:10]


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
    print(f"\n  [planner] 🗺 starting plan: '{goal}'", flush=True)

    try:
        # ── Phase 1: DECOMPOSE ────────────────────────────────────────────
        tool_names = ", ".join(_tool_registry.names()) or "none"
        decompose_prompt = "\n".join([
            "You are a task planner. Break the following goal into a numbered list",
            "of concrete, actionable steps.",
            "",
            "RULES for steps:",
            "1. Each step must be ONE of:",
            "   - A single tool call (write a file, fetch a URL, run a command, etc.)",
            "   - A single question answered from previous step results",
            "2. Each step must name the specific tool or action:",
            '   GOOD: "Write \'hello from Small O\' to ~/Desktop/notes.txt using write_file"',
            '   GOOD: "Fetch https://wikipedia.org text using fetch_url"',
            '   GOOD: "Read ~/Desktop/notes.txt using read_file"',
            '   BAD:  "Set the path"',
            '   BAD:  "Verify the result"',
            '   BAD:  "Notify the user"',
            "3. Use the exact URLs, file paths, and content from the original goal.",
            "   Do not substitute example.com or placeholder values.",
            "4. Do NOT include steps for things you will say or narrate.",
            "   Only include steps that require a tool call or computation.",
            "5. Maximum 6 steps. If you need more, combine related actions.",
            "",
            f"Available tools: {tool_names}",
            "",
            f"Goal: {goal}",
            "",
            "Respond ONLY with the numbered list. No explanation, no preamble.",
            "Format exactly:",
            "1. Step one text",
            "2. Step two text",
            "3. Step three text",
        ])
        decompose_resp = await _llm(decompose_prompt)
        steps = _parse_and_filter_steps(decompose_resp)

        if not steps:
            # LLM didn't produce a numbered list — treat the whole goal as one step
            steps = [goal]

        # Cap at max_steps
        steps = steps[:max_steps]

        print(f"  [planner] decomposed into {len(steps)} step(s)", flush=True)
        broadcast("PLAN_EVENT", {
            "phase": "decomposed",
            "steps": steps,
            "goal":  goal,
        })

        # ── Phase 2: EXECUTE ──────────────────────────────────────────────
        results: list[str] = []
        context_window: list[str] = []   # rolling last-3 results as context

        for i, step in enumerate(steps):
            print(f"  [planner] step {i+1}/{len(steps)}: {step[:80]}", flush=True)
            broadcast("PLAN_EVENT", {
                "phase":      "step_start",
                "step_index": i,
                "step_text":  step,
                "total":      len(steps),
            })

            try:
                context = "\n".join(context_window[-3:])
                result  = await _execute_step_with_tools(step, context)
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

            # Early exit: check if goal is already achieved
            if len(results) >= 2:   # need at least 2 results to be meaningful
                try:
                    done = await _check_goal_done(goal, results)
                    if done:
                        print(f"  [planner] ✓ goal achieved after {i+1} step(s)", flush=True)
                        break
                except Exception:
                    pass   # goal check failure is non-fatal

        # ── Phase 3: SUMMARIZE ────────────────────────────────────────────
        all_results = "\n".join(
            f"Step {j+1}: {r}" for j, r in enumerate(results)
        )
        summary_prompt = (
            f"{_SUMMARIZER_SYSTEM}\n\n"
            "Summarise what was accomplished for this goal in 2-3 sentences, "
            "speaking directly to the user in first person as their assistant.  "
            "Be concrete about what was done.  Voice output only — no lists or markdown.\n\n"
            f"Goal: {goal}\n\nResults:\n{all_results}"
        )
        summary = await asyncio.to_thread(ask_llm, summary_prompt)
        summary = summary.strip()
        # Safety net: strip any control tags that escaped the system-prompt guard
        import re as _re
        summary = _re.sub(r"<start_plan>.*?</start_plan>", "", summary, flags=_re.DOTALL).strip()
        summary = _re.sub(r"<tool_call>.*?</tool_call>",   "", summary, flags=_re.DOTALL).strip()

        print(f"  [planner] ✅ complete.  Summary: {summary[:100]}", flush=True)
        broadcast("PLAN_EVENT", {
            "phase":   "complete",
            "summary": summary,
            "goal":    goal,
        })

        # Emit LLM_TOKEN stream so TTS and the conversation UI pick up the summary.
        # Split into word-level tokens that match how the normal pipeline streams.
        broadcast("VOICE_STATE", {"state": "speaking"})
        words = summary.split()
        for j, word in enumerate(words):
            token = word if j == 0 else " " + word
            broadcast("LLM_TOKEN", {"token": token, "done": False})
            await asyncio.sleep(0.01)   # pace the UI render (10 ms per word)
        broadcast("LLM_TOKEN", {"token": "", "done": True, "token_count": len(words)})

        # Speak the summary aloud via TTS on a background thread so we
        # don't block the event loop.  We import tts here to avoid a
        # top-level circular import.
        try:
            from tts import speak as _speak
            stop_event = threading.Event()
            await asyncio.to_thread(_speak, summary, stop_event)
        except Exception as exc:
            print(f"  [planner] ⚠ TTS speak failed: {exc}", flush=True)

        broadcast("VOICE_STATE", {"state": "idle"})

    except asyncio.CancelledError:
        print("  [planner] ⛔ cancelled", flush=True)
        broadcast("PLAN_EVENT", {"phase": "cancelled", "goal": goal})
        broadcast("VOICE_STATE", {"state": "idle"})
        raise   # propagate so asyncio marks the task as cancelled

    except Exception as exc:
        import traceback
        reason = str(exc)
        print(f"  [planner] ✗ failed: {reason}", flush=True)
        traceback.print_exc()
        broadcast("PLAN_EVENT", {
            "phase":  "failed",
            "reason": reason,
            "goal":   goal,
        })
        # Speak the failure aloud so user is aware
        broadcast("LLM_TOKEN", {"token": f"I hit an error while working on that: {reason}", "done": False})
        broadcast("LLM_TOKEN", {"token": "", "done": True})
        broadcast("VOICE_STATE", {"state": "idle"})
