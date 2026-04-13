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
import json
import re
import threading
from typing import Callable

from tools.registry import registry as tool_registry

# LLM calls are synchronous (requests-based); we wrap them in asyncio.to_thread.
from llm.main_llm import ask_llm, _extract_tool_call, _run_tool_sync


# ── Regex helpers ─────────────────────────────────────────────────────────────

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\)]\s+(.+)$", re.MULTILINE)


def _parse_steps(text: str) -> list[str]:
    """Extract numbered step lines from an LLM decomposition response."""
    matches = _NUMBERED_LINE_RE.findall(text)
    return [m.strip() for m in matches if m.strip()]


# ── LLM helpers (all run via asyncio.to_thread) ───────────────────────────────

async def _llm(prompt: str, system: str = "") -> str:
    """One-shot async LLM call; returns full response text."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    return await asyncio.to_thread(ask_llm, full_prompt)


async def _execute_step_with_tools(step: str, context: str) -> str:
    """
    Ask the LLM to execute a single step, optionally using tools.

    Two-pass: collect full response, detect <tool_call>, run tool if found,
    then ask LLM to summarise the result.  Returns the final answer string.
    """
    system = (
        "You are Small O's autonomous execution engine.  "
        "Execute the given step concisely.  "
        "Use a tool if real-world data or action is needed.  "
        "Return only the result — no preamble, no explanation."
    )
    prompt = (
        f"Step to execute: {step}\n\n"
        f"Context from previous steps (most recent last):\n{context or 'None yet'}\n\n"
        "Execute the step.  If you need a tool, emit the tool_call block.  "
        "Otherwise answer directly."
    )

    raw = await asyncio.to_thread(ask_llm, f"{system}\n\n{prompt}")

    # Check for tool call in response
    tool_name, tool_args, visible = _extract_tool_call(raw)
    if tool_name is None:
        return raw.strip()

    # Dispatch tool (already async)
    print(f"  [planner] 🔧 tool: {tool_name}  args={tool_args}", flush=True)
    tool_result = await tool_registry.dispatch(tool_name, tool_args or {})
    print(f"  [planner] 🔧 result: {tool_result[:120]}", flush=True)

    # Second LLM pass: summarise tool result as the step outcome
    summary_prompt = (
        f"{system}\n\n"
        f"Step: {step}\n"
        f"Tool result:\n{tool_result}\n\n"
        "Summarise the result in one sentence as the step outcome."
    )
    summary = await asyncio.to_thread(ask_llm, summary_prompt)
    return summary.strip()


async def _check_goal_done(goal: str, results: list[str]) -> bool:
    """Ask the LLM: is the goal fully achieved?  Returns True on YES."""
    summary = "\n".join(f"- {r}" for r in results[-5:])   # last 5 results
    prompt = (
        f"Goal: {goal}\n\n"
        f"Results so far:\n{summary}\n\n"
        "Is the goal fully achieved?  Answer YES or NO and nothing else."
    )
    answer = await asyncio.to_thread(ask_llm, prompt)
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
        decompose_prompt = (
            "Break this goal into a numbered list of concrete steps.  "
            "Each step should be accomplishable with one tool call or one direct answer.  "
            "Respond ONLY with the numbered list.  No preamble, no explanation.\n\n"
            f"Goal: {goal}"
        )
        decompose_resp = await _llm(decompose_prompt)
        steps = _parse_steps(decompose_resp)

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
            "Summarise what was accomplished for this goal in 2-3 sentences, "
            "speaking directly to the user in first person as their assistant.  "
            "Be concrete about what was done.  Voice output only — no lists or markdown.\n\n"
            f"Goal: {goal}\n\nResults:\n{all_results}"
        )
        summary = await asyncio.to_thread(ask_llm, summary_prompt)
        summary = summary.strip()

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
