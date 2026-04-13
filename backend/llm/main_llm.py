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

from config.llm import LLM_CONFIG
from llm.SYSTEM_PROMPT import SYSTEM_PROMPT


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
    """Return the TOOLS section appended to the system prompt before each call."""
    reg = _get_registry()
    schemas = reg.get_schemas()
    if not schemas:
        return ""

    schema_json = json.dumps(schemas, indent=2)
    return (
        "\n\n## Available tools\n\n"
        "You can call tools by emitting a JSON block anywhere in your response "
        "in this exact format:\n\n"
        "<tool_call>\n"
        '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\"}}\n'
        "</tool_call>\n\n"
        "You may call one tool per response. After calling a tool you will "
        "receive the result and can continue your response.\n\n"
        f"Available tools:\n{schema_json}\n\n"
        "Rules:\n"
        "- Call a tool ONLY when you genuinely need it to answer the user.\n"
        "- Never fabricate tool results.\n"
        "- Always tell the user what you are about to do before the tool call.\n"
        "- After getting a result, summarise it naturally in your voice.\n"
        "- Never reveal internal tool names or raw JSON to the user.\n\n"
        "## Autonomous planner\n\n"
        "If the user's request requires MULTIPLE steps, multiple tool calls, "
        "or sustained background work (research, file creation, multi-stage "
        "computation), DO NOT attempt it in a single response.  Instead emit:\n\n"
        "<start_plan>\n"
        "one sentence description of the goal\n"
        "</start_plan>\n\n"
        "The autonomous planner will take over, execute every step, and report "
        "back when done.  For simple questions and single-action requests, "
        "respond directly as normal — do NOT trigger the planner unnecessarily."
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
    system_content = SYSTEM_PROMPT + _build_tools_section()
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

    response = requests.post(
        LLM_CONFIG.ollama_url,
        json={
            "model":      LLM_CONFIG.model,
            "messages":   messages,
            "stream":     True,
            "keep_alive": -1,
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
                "keep_alive": -1,
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

    # ── Pass 2: stream final answer with tool result in history ───────────
    # Inject: assistant message (preamble + tool call) + tool result
    history = [
        {"role": "assistant", "content": full_text},
        {"role": "tool",      "content": f"Tool '{tool_name}' result:\n{tool_result}"},
    ]
    messages2 = _build_messages(safe_text, system_suffix=system_suffix, extra_history=history)
    # Swap the trailing user message so it doesn't duplicate the original
    # question — instead ask the LLM to continue naturally from the result.
    messages2[-1] = {
        "role":    "user",
        "content": "Please summarise the tool result naturally and answer my request.",
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
        "content": "Please summarise the tool result naturally and answer my request.",
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

    # ── Otherwise: delegate to tool-or-plain handler (a generator) ─────────
    return _handle_tool_or_plain(full_text, tokens, safe_text, system_suffix)
