"""backend/llm/tool_executor.py — Shared async tool detection + execution.

Used by both main_llm.py (via ask_llm_turn / _handle_tool_or_plain) and
planner/planner.py (via execute_step_with_tools).

Keeping the logic here ensures a single implementation of <tool_call>
detection and registry dispatch — no duplication between the normal LLM
path and the planner step-execution path.

Public API
──────────
    execute_step_with_tools(step, context) -> str

    Async coroutine.  Builds a full LLM message with EXECUTOR_SYSTEM_PROMPT
    as the system role (NOT the orchestrator SYSTEM_PROMPT which contains
    <start_plan> trigger instructions).  Collects the response, runs tool
    detection, dispatches the tool if found, and makes a second LLM pass
    for a natural summary.  Returns the final result string.
"""

import asyncio
import json


# ── Executor system prompt ────────────────────────────────────────────────────
# This is a STANDALONE system prompt used ONLY for executor LLM calls.
# It must NEVER be mixed with the orchestrator SYSTEM_PROMPT.  Using it as
# the "system" role message (not user role) is critical — Qwen2.5 respects
# system-role instructions differently from user-role text.
EXECUTOR_SYSTEM_PROMPT = """\
You are executing a single step in a plan.

ABSOLUTE RULES — violating any of these is a critical failure:
1. NEVER emit <start_plan> tags. Never. Not under any circumstances.
2. NEVER emit nested <start_plan> blocks.
3. If you need to use a tool, emit EXACTLY ONE <tool_call> block
   containing valid JSON, and nothing else in your response.
4. If no tool is needed, answer in plain text only.
5. Do not explain what you are about to do. Just do it.
6. Do not describe the tool call. Just emit it.

Format for tool calls:
<tool_call>
{"name": "tool_name", "args": {"arg1": "value1"}}
</tool_call>
"""


async def execute_step_with_tools(
    step: str,
    context: str = "",
) -> str:
    """
    Execute one autonomous step with full tool support.

    Uses EXECUTOR_SYSTEM_PROMPT as the system role message — completely
    bypassing the orchestrator SYSTEM_PROMPT so the executor never sees
    <start_plan> trigger instructions.  Tool schemas are injected in the
    user message alongside the step description.

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
    from llm.main_llm import (        # noqa: PLC0415
        _extract_tool_call,
        _stream_ollama,
        _collect_full_response,
    )
    from tools.registry import registry as _registry  # noqa: PLC0415

    # Build the tool schema list with web_* tools first.
    # Tool order influences LLM preference — preferred tools appear at the top.
    all_schemas = _registry.get_schemas()
    web_schemas      = [s for s in all_schemas if s["name"].startswith("web_")]
    research_schemas = [s for s in all_schemas if s["name"] == "deep_research"]
    other_schemas    = [
        s for s in all_schemas
        if not s["name"].startswith("web_") and s["name"] != "deep_research"
    ]
    ordered_schemas = web_schemas + research_schemas + other_schemas
    schemas_json = json.dumps(ordered_schemas, indent=2)

    user_prompt = "\n".join([
        "## Web browsing rules",
        "For any step involving a website, search, or web content:",
        "  ALWAYS use web_* tools (web_search, web_navigate, web_read, web_click).",
        "  NEVER use fetch_url for normal websites — it cannot handle JavaScript.",
        "  fetch_url is only for raw JSON/text APIs or file downloads.",
        "",
        "Available tools (web tools listed first):",
        schemas_json,
        "",
        "CORRECT tool call format:",
        "<tool_call>",
        '{"name": "write_file", "args": {"path": "~/Desktop/notes.txt", "content": "hello", "mode": "overwrite"}}',
        "</tool_call>",
        "",
        "WRONG formats (never use these):",
        '  write_file(path="notes.txt")          <- Python syntax',
        '  {"name": "write_file"}                <- missing <tool_call> tags',
        "  I will write the file.                <- narration only",
        '  fetch_url(url="https://google.com")   <- use web_search instead',
        "",
        "If no tool is needed, respond with plain text only — no tags.",
        "",
        f"Step to execute: {step}",
        "",
        f"Context from previous steps:\n{context or 'None yet'}",
    ])

    # ── Pass 1: collect full LLM response ─────────────────────────────────────
    # Messages use EXECUTOR_SYSTEM_PROMPT as the system role — NOT _build_messages()
    # which would prepend the orchestrator prompt (with <start_plan> instructions).
    def _pass1():
        msgs = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
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

    # ── Tool call detected ─────────────────────────────────────────────────────
    print(f"  [executor] 🔧 tool: {tool_name}  args={tool_args}", flush=True)
    tool_result = await _registry.dispatch(tool_name, tool_args or {})
    print(f"  [executor] 🔧 result ({len(tool_result)} chars): {tool_result[:120]}", flush=True)

    # ── Pass 2: one-sentence natural-language summary of the tool result ───────
    # Still uses EXECUTOR_SYSTEM_PROMPT so the summary LLM call also can't emit
    # <start_plan> tags.
    def _pass2():
        msgs2 = [
            {"role": "system",    "content": EXECUTOR_SYSTEM_PROMPT},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": full_text},
            {"role": "tool",      "content": f"Tool '{tool_name}' result:\n{tool_result}"},
            {
                "role":    "user",
                "content": (
                    "Summarise the tool result in one sentence as the step outcome.  "
                    "Plain text only — no tags, no narration."
                ),
            },
        ]
        summary_text, _ = _collect_full_response(_stream_ollama(msgs2))
        return summary_text.strip()

    return await asyncio.to_thread(_pass2)
