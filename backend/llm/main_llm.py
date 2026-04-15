"""backend/llm/main_llm.py — LLM streaming client for Small O (Jarvis upgrade).

Architecture
────────────
This module owns the full LLM call lifecycle:

  1. Build a message list: system prompt + user-context + tool schemas + history
  2. Stream tokens from Ollama (qwen2.5:3b)
  3. After collecting the full response, scan for a <tool_call> JSON block
  4. If found: dispatch to ToolRegistry, append tool result, make a second
     streaming call for the final natural-language answer
  5. Yield every token from the final (or only) stream for TTS consumption

Two-pass approach (not single-pass streaming tool detection)
─────────────────────────────────────────────────────────────
Streaming tool detection sounds appealing but creates race conditions: the
LLM may emit the closing </tool_call> tag mid-stream before we know the full
JSON.  Two-pass is simpler, safer, and produces better TTS audio because the
final response stream starts immediately after tool execution with no partial
emission noise.

Phase 2 note: the ``messages`` list is the natural handoff point for a future
task-planner.  It can pre-populate the history with sub-agent results before
calling ask_llm_stream(), making multi-agent orchestration a simple extension.
"""

import asyncio
import json
import re
import threading
from typing import Iterator

import requests

from config.llm import LLM_CONFIG, KEEP_ALIVE_IDLE, KEEP_ALIVE_ACTIVE
from llm.SYSTEM_PROMPT import SYSTEM_PROMPT, get_runtime_context

# ── Conversation-active flag ──────────────────────────────────────────────────
# True  → model uses KEEP_ALIVE_ACTIVE (120 s) so it stays warm between turns.
# False → model uses KEEP_ALIVE_IDLE   (0 s) and evicts after each call.
# Set by main.py at turn start; cleared after 90 s of no input.
_conversation_active: bool = False


def set_conversation_active(active: bool) -> None:
    """Called by main.py to switch keep_alive tier for the 3b model."""
    global _conversation_active
    _conversation_active = active


# ── Tool registry (imported lazily to avoid circular imports at module load) ──
# We import inside functions so that tools/__init__.py can import main_llm
# without triggering a circular chain.  The first real LLM call will have
# already registered all tools via main.py's startup imports.

def _get_registry():
    from tools.registry import registry  # lazy import
    return registry


# ── Prompt-injection blocklist ────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"^you are ",
    r"act as ",
    r"pretend to be ",
    r"system prompt",
    r"developer message",
    r"ignore previous",
    r"follow these instructions",
]


def _sanitize_user_text(text: str) -> str:
    """Strip obvious prompt-injection attempts while preserving user intent."""
    clean_lines = []
    for line in text.splitlines():
        lower = line.strip().lower()
        if any(re.search(p, lower) for p in _INJECTION_PATTERNS):
            continue
        clean_lines.append(line)
    return " ".join(clean_lines).strip()


# ── Tool schema injection ─────────────────────────────────────────────────────

def _build_tools_section() -> str:
    """
    Return the TOOLS section appended to the system prompt before each call.

    Deliberately injects only tool *names + one-line descriptions* (not full
    parameter schemas).  Full schemas balloon the prompt by ~5 KB and reduce
    the effective weight of the mandatory-action rules near the top.  The LLM
    only needs to know which tools exist to pick one; parameter details are
    provided in the forced-tool-call fallback when needed.
    """
    reg = _get_registry()
    menu = reg.get_menu()
    if not menu:
        return ""

    menu_lines = "\n".join(f"  {t['name']}: {t['description']}" for t in menu)
    return (
        "\n\n## Available tools\n\n"
        "To call a tool, emit a <tool_call> block containing valid JSON with "
        "'name' matching one of the tools below and 'args' matching its parameters exactly.  "
        "Emit ONLY the <tool_call> block — no preamble, no explanation, nothing else.  "
        "The tool executes and returns a result; then you may speak in one sentence.\n\n"
        f"Tools:\n{menu_lines}\n\n"
        "## Autonomous planner\n\n"
        "Only emit a plan trigger when the request genuinely requires 3 or more "
        "distinct tool calls to complete.  Single-tool requests MUST be handled "
        "directly with a <tool_call> block — never with <start_plan>.\n\n"
        "Emit <start_plan> ONLY when ALL of these are true:\n"
        "- The task requires 3 or more distinct steps\n"
        "- Each step requires a separate tool call\n"
        "- The steps must happen in sequence (output of one feeds the next)\n\n"
        "NEVER emit <start_plan> for:\n"
        "- set_reminder (1 tool call)\n"
        "- web_search (1 tool call)\n"
        "- web_navigate (1 tool call)\n"
        "- Any single-tool action\n\n"
        "Plan trigger format (emit EXACTLY this, nothing else):\n\n"
        "<start_plan>\n"
        "one sentence description of the goal\n"
        "</start_plan>"
    )


def _build_forced_tool_system() -> str:
    """
    Minimal system prompt for the forced-tool-call fallback pass.

    Used when Pass 1 returns a conversational response despite the user
    clearly requesting an action.  Contains only the tool menu + strict
    instruction — no memory context, no persona, no clutter.
    """
    reg = _get_registry()
    menu_lines = "\n".join(
        f"  {t['name']}: {t['description']}" for t in reg.get_menu()
    )
    return (
        "You are a tool dispatcher.  The user wants to perform an action.\n"
        "You MUST respond with ONLY a <tool_call> block — no other text whatsoever.\n\n"
        "Format:\n"
        "<tool_call>\n"
        "{\"name\": \"tool_name\", \"args\": {\"arg\": \"value\"}}\n"
        "</tool_call>\n\n"
        f"Available tools:\n{menu_lines}"
    )


# ── Message builder ───────────────────────────────────────────────────────────

def _build_messages(
    user_text: str,
    system_suffix: str = "",
    extra_history: list[dict] | None = None,
) -> list[dict]:
    """
    Build the Ollama messages list for a single LLM call.

    Parameters
    ----------
    user_text:     The current user utterance (may already contain memory ctx).
    system_suffix: Additional text to append to the system prompt
                   (e.g., user_context snippet).
    extra_history: Additional messages to insert between system and user
                   (used for the second-pass tool-result injection).
    """
    system_content = SYSTEM_PROMPT + _build_tools_section() + "\n\n" + get_runtime_context()
    if system_suffix:
        system_content += "\n\n" + system_suffix

    # Decompose memory-context prefix from the utterance if present.
    # _build_memory_context in main.py formats the string as:
    #   "<context>\n\nUser: <utterance>"
    if "\n\nUser: " in user_text:
        memory_ctx, utterance = user_text.split("\n\nUser: ", 1)
        system_content += f"\n\n{memory_ctx}"
    else:
        utterance = user_text

    messages: list[dict] = [{"role": "system", "content": system_content}]

    if extra_history:
        messages.extend(extra_history)

    messages.append({"role": "user", "content": utterance})
    return messages


# ── Raw streaming call ────────────────────────────────────────────────────────

def _stream_ollama(messages: list[dict]) -> Iterator[str]:
    """Yield response tokens one by one from Ollama (blocking generator)."""
    total_chars = sum(len(m["content"]) for m in messages)
    print(f"  [llm] ▶ {total_chars:,} char prompt → {LLM_CONFIG.model}", flush=True)

    # Use ACTIVE keep_alive during a conversation so entity extraction and the
    # main LLM call share a warm model; idle turns still evict immediately.
    _keep_alive = KEEP_ALIVE_ACTIVE if _conversation_active else KEEP_ALIVE_IDLE
    response = requests.post(
        LLM_CONFIG.ollama_url,
        json={
            "model":      LLM_CONFIG.model,
            "messages":   messages,
            "stream":     True,
            "keep_alive": _keep_alive,
            "options": {
                "num_predict": LLM_CONFIG.num_predict,
                "stop":        ["User:", "Human:"],
            },
        },
        stream=True,
        timeout=(
            LLM_CONFIG.stream_timeout_connect_s,
            LLM_CONFIG.stream_timeout_read_s,
        ),
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done", False):
                yield data.get("message", {}).get("content", "")


def _collect_full_response(token_iter: Iterator[str]) -> tuple[str, list[str]]:
    """Drain the iterator, return (full_text, list_of_tokens)."""
    tokens: list[str] = []
    for tok in token_iter:
        tokens.append(tok)
    return "".join(tokens), tokens


# ── Tool-call and plan-trigger detection ─────────────────────────────────────

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

_PLAN_TRIGGER_RE = re.compile(
    r"<start_plan>\s*(.+?)\s*</start_plan>",
    re.DOTALL | re.IGNORECASE,
)

# ── Multi-step fallback heuristics ─────────────────────────────────────────────
# If the LLM forgets to emit <start_plan>, these patterns in the user's text are
# a strong signal the task is multi-step.  Matching 2+ patterns forces a trigger.
_MULTI_STEP_PATTERNS: list = [
    re.compile(r'\bthen\b',          re.IGNORECASE),
    re.compile(r'\band\s+then\b',    re.IGNORECASE),
    re.compile(r'\bafter\s+that\b',  re.IGNORECASE),
    re.compile(r'\bnext[,\s]',       re.IGNORECASE),
    re.compile(r'\bfinally\b',       re.IGNORECASE),
    re.compile(r'\bstep\s+\d\b',     re.IGNORECASE),
    re.compile(r'\bfirst\b.{1,60}\bthen\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'\balso\b.{1,60}\b(create|write|fetch|read|run|open)\b',
               re.IGNORECASE | re.DOTALL),
]

# ── Single-action tool-required patterns ──────────────────────────────────────
# When the LLM gives a conversational response but the user's request clearly
# requires ONE tool call, these patterns trigger a second minimal forced-tool
# LLM call that discards the conversational response and returns the tool result.
# Each pattern must match the FULL user utterance (after stripping memory ctx).
_TOOL_REQUIRED_PATTERNS: list[re.Pattern] = [
    # Web navigation / opening
    re.compile(
        r'\b(go\s+to|open|navigate\s+to|visit|load)\b.*(\.com|\.org|\.net|\.io|\.co|http)',
        re.IGNORECASE,
    ),
    # Web search
    re.compile(
        r'\b(search|look\s+up|find|google)\b.+\b(for|about|on)\b',
        re.IGNORECASE,
    ),
    # Web content read / fetch
    re.compile(
        r'\b(read|fetch|get|scrape)\b.*(page|site|url|http|\.com)',
        re.IGNORECASE,
    ),
    # Web interaction
    re.compile(r'\b(click|type\s+in|fill\s+in|submit)\b', re.IGNORECASE),
    # File operations
    re.compile(
        r'\b(create|write|save|read)\b.*(file|\.txt|\.md|\.py|\.json|\.csv)',
        re.IGNORECASE,
    ),
    # Terminal / shell
    re.compile(r'\b(run|execute|open\s+terminal)\b', re.IGNORECASE),
    # Reminders
    re.compile(r'\b(remind\s+me|set\s+a\s+reminder|alert\s+me)\b', re.IGNORECASE),
]


def _extract_tool_call(text: str) -> tuple[str | None, dict | None, str]:
    """
    Scan text for a <tool_call>...</tool_call> block.

    Returns (name, args, visible_text) where visible_text has the block
    stripped out.  If no block is found, name and args are None.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None, None, text

    try:
        payload = json.loads(m.group(1))
        name: str   = payload["name"]
        args: dict  = payload.get("args", {})
    except (json.JSONDecodeError, KeyError):
        # Malformed block — treat as plain text
        return None, None, text

    # Strip the block from the visible portion
    visible = _TOOL_CALL_RE.sub("", text).strip()
    return name, args, visible


# ── Async tool dispatch bridge ────────────────────────────────────────────────

def _run_tool_sync(name: str, args: dict) -> str:
    """
    Execute an async tool handler from the synchronous LLM thread.

    The pipeline thread is NOT the asyncio event loop thread.  We can't
    ``await`` here, so we run a fresh event loop on the pipeline thread just
    for this one dispatch.  The reminder tool's asyncio.ensure_future() call
    DOES need the main event loop — we handle that special case by scheduling
    the coroutine on the main loop via run_coroutine_threadsafe.
    """
    reg = _get_registry()

    # Try to get the main running loop so reminder tasks land on it
    try:
        import backend_loop_ref  # set by main.py
        main_loop = backend_loop_ref.loop
    except (ImportError, AttributeError):
        main_loop = None

    if main_loop is not None and main_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(reg.dispatch(name, args), main_loop)
        try:
            return future.result(timeout=60)
        except Exception as exc:
            return f"Error running tool '{name}': {exc}"
    else:
        # Fallback: own event loop (works for all tools except reminder asyncio tasks)
        return asyncio.run(reg.dispatch(name, args))


# ── Forced tool-call fallback ─────────────────────────────────────────────────

def _forced_tool_call_stream(
    original_utterance: str,
    safe_text: str,
    system_suffix: str,
) -> "Iterator[str] | None":
    """
    Make a minimal second LLM call that is almost certain to emit a tool call.

    Called when Pass 1 returned a conversational response but the user's
    request matches one of the _TOOL_REQUIRED_PATTERNS.  The prompt is stripped
    to just system-instruction + tool menu + the raw user request — no memory
    context, no persona padding — so the small model has no excuse to waffle.

    Returns a token iterator (same type as normal response) on success,
    or None if the focused call also fails to produce a tool call (rare —
    caller falls back to the original conversational tokens).
    """
    messages = [
        {"role": "system", "content": _build_forced_tool_system()},
        {"role": "user",   "content": original_utterance},
    ]
    print(
        f"  [llm] 🔁 forced tool-call pass  (utterance: {original_utterance[:60]!r})",
        flush=True,
    )
    full_text, tokens = _collect_full_response(_stream_ollama(messages))

    tool_name, tool_args, _ = _extract_tool_call(full_text)
    if tool_name is None:
        print(
            f"  [llm] ⚠ forced tool-call pass produced no <tool_call>: "
            f"{full_text[:120]!r}",
            flush=True,
        )
        return None

    print(
        f"  [llm] 🔧 forced tool call: {tool_name}  args={tool_args}",
        flush=True,
    )
    # Re-use the standard two-pass handler with a synthetic full_text
    synthetic = (
        f'<tool_call>{{"name": "{tool_name}", "args": {json.dumps(tool_args or {})}}}'
        f"</tool_call>"
    )
    return _handle_tool_or_plain(synthetic, [synthetic], safe_text, system_suffix)


# ── Public interface ──────────────────────────────────────────────────────────

def warmup() -> None:
    """Probe Ollama, confirm model is loaded, seed the KV cache."""
    if not check_ollama():
        print("  [llm] ⚠  Ollama not ready — first turn may be slow", flush=True)
        return
    try:
        requests.post(
            LLM_CONFIG.ollama_url,
            json={
                "model":    LLM_CONFIG.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": "Hi."},
                ],
                "stream":     False,
                "keep_alive": 0,   # don't hold RAM after warmup probe
                "options":    {"num_predict": 1},
            },
            timeout=30,
        )
        print(f"  [llm] ✓ model '{LLM_CONFIG.model}' warmed up", flush=True)
    except Exception as e:
        print(f"  [llm] ⚠  warmup request failed: {e}", flush=True)


def check_ollama() -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code != 200:
            print(f"  [llm] ✗ Ollama health check failed (HTTP {r.status_code})", flush=True)
            return False
        tags  = r.json().get("models", [])
        names = [m.get("name", "") for m in tags]
        if not any(LLM_CONFIG.model in n for n in names):
            print(
                f"  [llm] ✗ Model '{LLM_CONFIG.model}' not in Ollama. "
                f"Available: {names}  →  run: ollama pull {LLM_CONFIG.model}",
                flush=True,
            )
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(
            f"  [llm] ✗ Cannot connect to Ollama — is it running?  (ollama serve)",
            flush=True,
        )
        return False
    except Exception as e:
        print(f"  [llm] ✗ Ollama health check error: {e}", flush=True)
        return False


def ask_llm(user_text: str, system_suffix: str = "") -> str:
    """Single-shot (non-streaming) LLM call.  Returns full response string."""
    safe_text = _sanitize_user_text(user_text)
    messages  = _build_messages(safe_text, system_suffix=system_suffix)
    full, _   = _collect_full_response(_stream_ollama(messages))
    return full.strip()


def ask_llm_stream(user_text: str, system_suffix: str = "") -> Iterator[str]:
    """
    Streaming LLM call with transparent tool-call detection.

    Yields response tokens for TTS/frontend consumption.  Internally:
      Pass 1 — collect full response, scan for <tool_call>.
      If found:
        • Execute the tool synchronously.
        • Append tool result to history.
        • Pass 2 — stream the final natural-language answer.
      If not found:
        • Re-yield the already-collected tokens.
    """
    safe_text = _sanitize_user_text(user_text)
    messages  = _build_messages(safe_text, system_suffix=system_suffix)

    # ── Pass 1: collect full response ──────────────────────────────────────
    full_text, tokens = _collect_full_response(_stream_ollama(messages))

    tool_name, tool_args, visible_text = _extract_tool_call(full_text)

    if tool_name is None:
        # No tool call — re-yield collected tokens directly
        yield from tokens
        return

    # ── Tool call detected ─────────────────────────────────────────────────
    print(f"  [llm] 🔧 tool call detected: {tool_name}  args={tool_args}", flush=True)

    # Yield the visible preamble before the tool block so TTS can start early
    if visible_text:
        yield visible_text + " "

    # Execute tool
    tool_result = _run_tool_sync(tool_name, tool_args or {})
    print(f"  [llm] 🔧 tool result ({len(tool_result)} chars): {tool_result[:120]}", flush=True)

    # ── Pass 2: stream brief final answer with tool result in history ─────
    # The user can see the browser / file / terminal output directly, so the
    # verbal response should be at most one sentence of confirmation.
    history = [
        {"role": "assistant", "content": full_text},
        {"role": "tool",      "content": f"Tool '{tool_name}' result:\n{tool_result}"},
    ]
    messages2 = _build_messages(safe_text, system_suffix=system_suffix, extra_history=history)
    messages2[-1] = {
        "role":    "user",
        "content": (
            "The tool just executed and the user can see the result directly "
            "(browser is visible, file was written, etc.). "
            "Respond in ONE sentence only — briefly confirm what happened or "
            "add one useful observation.  Do NOT repeat the user's request.  "
            "Do NOT describe what you did step by step."
        ),
    }
    print(f"  [llm] ▶ pass-2 stream after tool '{tool_name}'", flush=True)
    yield from _stream_ollama(messages2)


# ── Internal helper: tool or plain token stream ───────────────────────────────

def _handle_tool_or_plain(
    full_text: str,
    tokens: list[str],
    safe_text: str,
    system_suffix: str,
) -> Iterator[str]:
    """Generator: either re-yields plain tokens or does the two-pass tool flow."""
    tool_name, tool_args, visible_text = _extract_tool_call(full_text)

    if tool_name is None:
        yield from tokens
        return

    print(f"  [llm] 🔧 tool call detected: {tool_name}  args={tool_args}", flush=True)
    if visible_text:
        yield visible_text + " "

    tool_result = _run_tool_sync(tool_name, tool_args or {})
    print(f"  [llm] 🔧 tool result ({len(tool_result)} chars): {tool_result[:120]}", flush=True)

    history = [
        {"role": "assistant", "content": full_text},
        {"role": "tool",      "content": f"Tool '{tool_name}' result:\n{tool_result}"},
    ]
    messages2 = _build_messages(safe_text, system_suffix=system_suffix, extra_history=history)
    messages2[-1] = {
        "role":    "user",
        "content": (
            "The tool just executed and the user can see the result directly "
            "(browser is visible, file was written, etc.). "
            "Respond in ONE sentence only — briefly confirm what happened or "
            "add one useful observation.  Do NOT repeat the user's request.  "
            "Do NOT describe what you did step by step."
        ),
    }
    print(f"  [llm] ▶ pass-2 stream after tool '{tool_name}'", flush=True)
    yield from _stream_ollama(messages2)


def ask_llm_turn(
    user_text: str,
    system_suffix: str = "",
) -> "dict | Iterator[str]":
    """
    Combined LLM turn function used by _run_turn in backend/main.py.

    Returns EITHER:
      • {"type": "plan_trigger", "goal": str}   — if <start_plan> detected
      • Iterator[str] of tokens                 — normal response (with tool
                                                  handling transparent)

    Unlike ask_llm_stream (which is for backward compat / plugin summariser),
    this function additionally detects the <start_plan> planner trigger and
    returns a sentinel dict so _run_turn can start the autonomous planner
    without modifying the token stream.

    Priority: plan_trigger > tool_call > plain text
    """
    safe_text = _sanitize_user_text(user_text)
    messages  = _build_messages(safe_text, system_suffix=system_suffix)

    # Pass 1: collect the full response before making any decisions
    full_text, tokens = _collect_full_response(_stream_ollama(messages))

    # ── Plan trigger takes highest priority ────────────────────────────────
    plan_match = _PLAN_TRIGGER_RE.search(full_text)
    if plan_match:
        goal = plan_match.group(1).strip()
        print(f"  [llm] 🗺 plan trigger: '{goal}'", flush=True)
        return {"type": "plan_trigger", "goal": goal}

    # Extract the bare user utterance (strip memory-context prefix if present)
    utterance = (
        safe_text.split("\n\nUser: ", 1)[-1].strip()
        if "\n\nUser: " in safe_text
        else safe_text.strip()
    )

    # ── Fallback multi-step detection (safety net) ─────────────────────────
    # If the LLM forgot to emit <start_plan> AND produced no tool call,
    # check whether the user's input text strongly signals a multi-step task.
    tool_name_check, _, _ = _extract_tool_call(full_text)
    if tool_name_check is None:
        match_count = sum(1 for p in _MULTI_STEP_PATTERNS if p.search(safe_text))
        if match_count >= 2:
            print(
                f"  [llm] ⚠ forced plan trigger — LLM forgot to emit one "
                f"({match_count} multi-step patterns matched in user input)",
                flush=True,
            )
            return {"type": "plan_trigger", "goal": utterance}

    # ── Single-action tool-required fallback ───────────────────────────────
    # If the LLM gave a conversational response AND no tool call was detected
    # AND the user's utterance matches a tool-required pattern, force a second
    # minimal LLM call that reliably extracts the correct tool call.
    tool_name_check2, _, _ = _extract_tool_call(full_text)
    if tool_name_check2 is None:
        if any(p.search(utterance) for p in _TOOL_REQUIRED_PATTERNS):
            print(
                f"  [llm] ⚠ no tool call but utterance matches tool-required pattern "
                f"— running forced tool-call pass",
                flush=True,
            )
            forced = _forced_tool_call_stream(utterance, safe_text, system_suffix)
            if forced is not None:
                return forced
            # Forced pass also failed — fall through to conversational response
            print("  [llm] ⚠ forced tool-call pass failed — using conversational response", flush=True)

    # ── Otherwise: delegate to tool-or-plain handler (a generator) ─────────
    return _handle_tool_or_plain(full_text, tokens, safe_text, system_suffix)
