"""backend/llm/tool_executor.py — Shared async tool detection + execution.

Used by both main_llm.py (via ask_llm_turn / _handle_tool_or_plain) and
planner/planner.py (via execute_step_with_tools).

Keeping the logic here ensures a single implementation of <tool_call>
detection and registry dispatch — no duplication between the normal LLM
path and the planner step-execution path.

Public API
──────────
    execute_step_with_tools(step, context) -> str

    Async coroutine.  Builds a full LLM message (with tool schemas, via
    main_llm._build_messages), collects the response, runs tool detection,
    dispatches the tool if found, and makes a second LLM pass for a natural
    summary.  Returns the final result string.
"""

import asyncio
import json


async def execute_step_with_tools(
    step: str,
    context: str = "",
) -> str:
    """
    Execute one autonomous step with full tool support.

    Internally uses the same LLM message-building pipeline as the normal
    turn path (system prompt + tool schemas via _build_messages), so the
    LLM always sees identical tool-call instructions regardless of whether
    it is running inside a planner or a regular turn.

    Parameters
    ----------
    step:     One-sentence description of the step to execute.
    context:  Newline-joined summaries of previous step results (may be "").

    Returns
    -------
    Result text — either the LLM's direct answer (no tool needed) or the
    LLM's one-sentence summary of the tool result.
    """
    # Import inside function to avoid circular imports at module load time.
    # (planner imports this module; main_llm imports planner indirectly.)
    from llm.main_llm import (        # noqa: PLC0415
        _extract_tool_call,
        _build_messages,
        _stream_ollama,
        _collect_full_response,
    )
    from tools.registry import registry as _registry  # noqa: PLC0415

    # Embed tool schemas + a concrete format example directly in the prompt.
    # The LLM has been observed to produce Python-style calls (write_file(...))
    # when it only sees the schema list without an explicit usage example.
    # Showing one CORRECT and three WRONG examples prevents this reliably.
    schemas_json = json.dumps(_registry.get_schemas(), indent=2)

    user_prompt = "\n".join([
        "You are executing one step of an autonomous plan.",
        "",
        "Available tools:",
        schemas_json,
        "",
        "To use a tool you MUST emit EXACTLY this format — and nothing else:",
        "",
        "<tool_call>",
        '{"name": "tool_name_here", "args": {"arg1": "value1", "arg2": "value2"}}',
        "</tool_call>",
        "",
        "CORRECT example:",
        "<tool_call>",
        '{"name": "write_file", "args": {"path": "~/Desktop/notes.txt", "content": "hello", "mode": "overwrite"}}',
        "</tool_call>",
        "",
        "WRONG — do not use any of these formats:",
        '  write_file(path="notes.txt")          <- Python syntax, WRONG',
        '  {"name": "write_file"}                <- missing <tool_call> tags, WRONG',
        "  I will write the file for you.        <- narration only, WRONG",
        "",
        "If no tool is needed to answer this step, respond with plain text only — no tags.",
        "",
        f"Step to execute: {step}",
        "",
        f"Context from previous steps (most recent last):\n{context or 'None yet'}",
    ])

    # system_suffix appends after SYSTEM_PROMPT + tool schemas — keeps the
    # executor framing without replacing the base persona.
    #
    # CRITICAL rules injected here because the base system prompt still
    # includes planner-trigger instructions that the executor must override:
    #   • <start_plan> tags are FORBIDDEN — a plan is already running.
    #   • Only one <tool_call> block is permitted per step.
    #   • No narration — emit the tool call block or plain text, nothing else.
    system_suffix = (
        "CRITICAL: You are executing a single step inside an active plan.  "
        "NEVER emit <start_plan> tags — you are already inside a plan and emitting "
        "one will break the execution loop.  "
        "NEVER emit <tool_call> tags for anything other than the one tool you are "
        "actually calling right now.  "
        "If you need to use a tool, emit exactly one <tool_call> block and nothing else.  "
        "If no tool is needed, answer directly in plain text.  "
        "One tool call maximum per step.  Do NOT narrate or explain — only the result."
    )

    # ── Pass 1: collect full LLM response ────────────────────────────────────
    def _pass1():
        msgs = _build_messages(user_prompt, system_suffix=system_suffix)
        return _collect_full_response(_stream_ollama(msgs))

    full_text, _tokens = await asyncio.to_thread(_pass1)
    print(
        f"  [executor] step LLM response ({len(full_text)} chars): "
        f"{full_text[:100]!r}",
        flush=True,
    )

    tool_name, tool_args, _visible = _extract_tool_call(full_text)

    if tool_name is None:
        # No tool call — return the LLM's direct answer
        return full_text.strip()

    # ── Tool call detected ────────────────────────────────────────────────────
    print(f"  [executor] 🔧 tool: {tool_name}  args={tool_args}", flush=True)
    tool_result = await _registry.dispatch(tool_name, tool_args or {})
    print(f"  [executor] 🔧 result ({len(tool_result)} chars): {tool_result[:120]}", flush=True)

    # ── Pass 2: natural one-sentence summary of the tool result ──────────────
    def _pass2():
        history = [
            {"role": "assistant", "content": full_text},
            {"role": "tool",      "content": f"Tool '{tool_name}' result:\n{tool_result}"},
        ]
        msgs2 = _build_messages(
            user_prompt,
            system_suffix=system_suffix,
            extra_history=history,
        )
        msgs2[-1] = {
            "role":    "user",
            "content": "Summarise the tool result in one sentence as the step outcome.",
        }
        summary_text, _ = _collect_full_response(_stream_ollama(msgs2))
        return summary_text.strip()

    return await asyncio.to_thread(_pass2)
