"""backend/main.py – WebSocket server + voice pipeline.

Architecture: Continuous STT + VAD as Timestamp Oracle
───────────────────────────────────────────────────────

  Microphone
      │
      ├──► RollingAudioBuffer (60 s circular, mic NEVER gated)
      │             │
      │             └──► VADOracle (parallel, fires timestamps only)
      │                       │
      │             on_speech_start(t=18.5s)  ← barge-in trigger
      │             on_speech_end(18.5s, 22.3s)
      │                       │
      └──── buffer.read_window(18.5, 22.3) → audio_data
                                    │
                               Whisper → transcript → LLM → TTS

Why this fixes first-word clipping
────────────────────────────────────
OLD (VAD-gated): VAD onset fires ~32 ms into speech → first word partially
  clipped → "what is your name" → "it is your name".

NEW: Audio is ALWAYS in the 60 s buffer.  VADOracle subtracts pre_buffer_s
  (default 2.0 s) from the onset timestamp, so read_window() retrieves audio
  from 2 s BEFORE the first word was detected — guaranteed complete.

Barge-in flow
─────────────
  VADOracle fires on_speech_start while _current_voice_state == "speaking"
  → abort_speaking() immediately (no delay)
  → on_speech_end fires → _speech_events.put((start_s, end_s))
  → new _run_turn reads event → buffer.read_window() → Whisper

Events emitted (JSON):
  VOICE_STATE      {state: idle | listening | thinking | speaking}
  STT_RESULT       {text, recording_time, transcription_time}
  LLM_TOKEN        {token, done}
  PLUGIN_ACTION    {plugin, action, result, direct}
  MEMORY_EVENT     {type, importance, summary, id, retrieved?}
  SYSTEM_STATS     {cpu, ram, battery}
  PROACTIVE_EVENT  {event: "reminder", message: str}  ← fired by reminder_tool

Events received (JSON):
  ping                  → pong
  UPDATE_USER_CONTEXT   {key: str, value: any}  ← update persistent user model
  TEXT_INPUT            {text: str}             ← typed message, bypasses STT
"""
import asyncio
import json
import queue
import re
import sys
import threading
import time
from collections import Counter, deque

from pathlib import Path

import math

import numpy as np
import psutil
import websockets

# High-quality audio resampling — scipy is always available (scikit-learn dep).
# resample_poly uses a polyphase FIR filter with anti-aliasing, far better than
# np.interp (linear interpolation has no anti-alias filter and folds frequencies
# > target_SR/2 back into the audio band as noise, confusing Whisper).
try:
    from scipy.signal import resample_poly as _resample_poly
    def _resample(x: np.ndarray, src_sr: int, dst_sr: int = 16_000) -> np.ndarray:
        if src_sr == dst_sr:
            return x
        g    = math.gcd(src_sr, dst_sr)
        up   = dst_sr  // g
        down = src_sr  // g
        return _resample_poly(x, up, down).astype(np.float32)
except ImportError:
    # Fallback: linear interpolation (lower quality but always works)
    def _resample(x: np.ndarray, src_sr: int, dst_sr: int = 16_000) -> np.ndarray:
        if src_sr == dst_sr:
            return x
        n = int(len(x) * dst_sr / src_sr)
        return np.interp(
            np.linspace(0, len(x), n), np.arange(len(x)), x
        ).astype(np.float32)

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from audio import RollingAudioBuffer
from stt import transcribe, transcribe_partial, warmup as stt_warmup, StreamingTranscriber
from vad import VADOracle
from llm import ask_llm_stream, warmup as llm_warmup
from llm.main_llm import ask_llm_turn
from tts import speak, speak_stream, warmup as tts_warmup, abort_speaking
from tts.main_tts import register_ws_audio_sender
from memory_system.retrieval.search import retrieve_memories
from memory_system.core.insert_pipeline import insert_memory
from plugins.router import PluginRouter
from utils.latency import LatencyTracker

# ── Jarvis upgrade: tools + user context ──────────────────────────────────────
import backend_loop_ref                                  # loop ref for tool dispatch
import tools                                             # triggers self-registration of all four tools  # noqa: F401
from tools.reminder_tool import set_broadcast_fn as _set_reminder_broadcast, shutdown_all_reminders
from user_context import load_user_context, get_context_prompt, update_user_context

# ── Text input watcher ────────────────────────────────────────────────────────
from stt.text_input import watch_text_input

# ── Phase 2: autonomous planner ───────────────────────────────────────────────
from planner.planner import run_plan
from utils.ram_monitor import get_available_ram_gb, get_memory_pressure, can_load_7b
from config.llm import LLM_CONFIG
from llm.main_llm import set_conversation_active

# ── Phase 3: web agent + deep research ───────────────────────────────────────
import web_agent                                          # triggers tool self-registration  # noqa: F401
from web_agent.agent import set_broadcast_fn as _set_web_broadcast
from web_agent.monitor import web_monitor
from web_agent.browser import BrowserManager
import adapters.gpt_researcher_adapter                   # registers deep_research tool  # noqa: F401
from adapters.gpt_researcher_adapter import set_broadcast_fn as _set_research_broadcast
import adapters.browser_use_adapter                      # registers web_task tool  # noqa: F401
from adapters.browser_use_adapter import set_broadcast_fn as _set_browser_use_broadcast
import tools.close_heavy_tabs                            # registers close_heavy_tabs tool  # noqa: F401
import tools.memory_tool                                 # registers clear_memory tool     # noqa: F401

import httpx as _httpx
import uuid as _uuid


# ──────────────────────────────────────────────────
# Session ID — unique per server startup
# ──────────────────────────────────────────────────

SESSION_ID: str = _uuid.uuid4().hex[:12]


# ──────────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_clients: set = set()

# Raw audio bytes from browser  [uint32 SR][float32[] samples]
_audio_queue: queue.Queue = queue.Queue()

# Speech event timestamps (start_s, end_s) from VADOracle → pipeline
_speech_events: queue.Queue = queue.Queue()

# Text typed by the user (via WS TEXT_INPUT or file watcher) → pipeline.
# Each item is a plain str.  Checked before speech events in _run_turn.
_text_input_queue: queue.Queue = queue.Queue()

# Planner → pipeline signalling for plan-trigger responses.
# Planner puts either the string "MULTI" (multi-step plan, say "On it") or
# a result string (single-tool completed, speak the result directly).
# The pipeline thread blocks on this queue after launching a plan.
_plan_result_queue: queue.Queue = queue.Queue(maxsize=1)

# Active autonomous planner task.  Only one plan runs at a time;
# starting a new one cancels the previous.
_current_plan_task: asyncio.Task | None = None

# Continuous 60-second audio buffer — mic NEVER gated; set in _audio_ingestion_loop
_audio_buffer: RollingAudioBuffer | None = None

# Set by VAD oracle on barge-in; cleared by pipeline at start of each speaking phase
_interrupt_event: threading.Event = threading.Event()

# Current voice state — cached so new WS clients get it immediately
_current_voice_state: str = "idle"

# TTS enabled flag — toggled by frontend SET_TTS_ENABLED message.
# When False: LLM still runs and tokens stream to frontend; audio synthesis skipped.
_tts_enabled: bool = True

# Partial bot response saved when user barge-ins mid-sentence.
# Injected into the next turn's LLM prompt so the bot knows where it left off.
_interrupted_partial: str = ""

# ── Conversation-active tracking (drives 3-tier keep_alive in main_llm) ──────
# Updated to current time whenever a real turn processes user input.
# After 90 s of silence the model switches back to KEEP_ALIVE_IDLE (0 s).
_last_turn_time: float = 0.0
_CONVERSATION_IDLE_S: float = 90.0

# Protects _interrupted_partial across VAD callback and pipeline threads
_barge_in_lock: threading.Lock = threading.Lock()

# ── Planner management helpers ────────────────────────────────────────────────

# Words/phrases that cancel the running plan — checked BEFORE LLM call (fast).
_CANCEL_WORDS = frozenset([
    "stop", "cancel", "abort",
    "stop the task", "cancel the plan", "stop what you're doing",
    "stop that", "cancel that", "never mind",
])

# ── Turn-in-progress tracking (FIX1A / FIX1C watchdog) ───────────────────────
# Set True/timestamp when _run_turn starts; False when it returns (any path).
# The _pipeline_watchdog asyncio task reads these to detect stuck turns (>90s)
# and resets VOICE_STATE to idle so the user isn't left in silence forever.
_turn_in_progress: bool = False
_turn_started_at: float = 0.0

# ── Short-term recent-actions context buffer ──────────────────────────────────
# Holds the last 5 planner completions / tool results so the LLM can answer
# follow-up questions ("where did you save that?") without relying on semantic
# memory retrieval which may not rank the action highly enough.
_recent_actions: deque = deque(maxlen=5)


def add_recent_action(action: str) -> None:
    """Add a recent action string to the short-term buffer.  Thread-safe (GIL)."""
    _recent_actions.append(action)


def get_recent_actions_prompt() -> str:
    """Return formatted recent-actions text for LLM system-prompt injection."""
    if not _recent_actions:
        return ""
    lines = "\n".join(f"- {a}" for a in _recent_actions)
    return f"## Recent actions\n{lines}"


async def _launch_plan(goal: str) -> None:
    """Start a new plan task on the main event loop, cancelling any active plan."""
    global _current_plan_task
    if _current_plan_task and not _current_plan_task.done():
        _current_plan_task.cancel()
        try:
            await _current_plan_task
        except asyncio.CancelledError:
            pass
    # Pass result_queue and action_callback directly — avoids the circular-import
    # bug where planner's ``from main import _plan_result_queue`` imports main.py
    # as a fresh module (not __main__), creating a different queue object that the
    # pipeline thread never reads.  (FIX1B root cause fix)
    _current_plan_task = asyncio.create_task(
        run_plan(goal, _emit, _plan_result_queue, add_recent_action, max_steps=20),
        name=f"plan:{goal[:40]}",
    )
    print(f"  [plan] task created: {_current_plan_task.get_name()}", flush=True)

    # After the plan finishes (success or failure), pre-warm 3b model so the
    # next conversation turn doesn't wait for a cold model load (120 s window).
    def _on_plan_done(task: asyncio.Task) -> None:
        asyncio.ensure_future(_preload_model(LLM_CONFIG.model, keep_alive_s=120))

    _current_plan_task.add_done_callback(_on_plan_done)


async def _cancel_plan() -> None:
    """Cancel the active plan task if one is running."""
    global _current_plan_task
    if _current_plan_task and not _current_plan_task.done():
        _current_plan_task.cancel()
        try:
            await _current_plan_task
        except asyncio.CancelledError:
            pass
        print("  [plan] task cancelled by user", flush=True)
    _current_plan_task = None


# ── Predictive model preloader ────────────────────────────────────────────────

async def _preload_model(model: str, keep_alive_s: int = 60) -> None:
    """
    Warm up an Ollama model by sending a 1-token dummy request.

    Uses keep_alive="{keep_alive_s}s" so the model stays warm for one
    typical turn cycle (default 60 s) without hogging RAM indefinitely.

    Called as a fire-and-forget background task at:
      • Server startup (after warmup completes)
      • End of each plan (pre-load 3b so next conversation turn is instant)
      • STT start (while Whisper is running, model may warm up)

    Non-blocking — failures are logged but never raised.
    """
    from config.llm import LLM_CONFIG as _cfg  # noqa: PLC0415
    try:
        async with _httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                _cfg.ollama_url,
                json={
                    "model":      model,
                    "messages":   [{"role": "user", "content": "hi"}],
                    "stream":     False,
                    "keep_alive": f"{keep_alive_s}s",
                    "options":    {"num_predict": 1},
                },
            )
        print(f"  [llm] preloaded {model} (keep_alive={keep_alive_s}s)", flush=True)
    except Exception as exc:
        print(f"  [llm] preload {model} failed (non-fatal): {exc}", flush=True)


# Streaming STT — one StreamingTranscriber per utterance.
# It runs Whisper every 500 ms during speech (live word-by-word display)
# and produces the final result in finalize() at turn end.
def _make_transcriber() -> StreamingTranscriber:
    """Create a fresh StreamingTranscriber that emits STT_PARTIAL events."""
    def _on_partial(confirmed: str, hypothesis: str) -> None:
        _emit("STT_PARTIAL", {"text": confirmed, "hypothesis": hypothesis})
    return StreamingTranscriber(
        transcribe_fn         = transcribe,
        on_partial            = _on_partial,
        chunk_interval_s      = 0.3,
        transcribe_partial_fn = transcribe_partial,
    )

_active_transcriber: list = [None]   # [StreamingTranscriber | None]


def _emit(event: str, data: dict):
    """Thread-safe broadcast to all connected WebSocket clients."""
    global _current_voice_state
    if event == "VOICE_STATE":
        _current_voice_state = data.get("state", _current_voice_state)
    if not _loop or not _clients:
        return

    async def _send_all():
        msg  = json.dumps({"event": event, "data": data})
        dead = set()
        for ws in list(_clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        _clients.difference_update(dead)

    asyncio.run_coroutine_threadsafe(_send_all(), _loop)


def _make_audio_sender():
    """
    Build the WebSocket audio sender callback for main_tts.register_ws_audio_sender().

    The callback accepts two mutually exclusive call signatures:
      fn(msg_dict, None)    → broadcast a JSON event to all clients
      fn(None, raw_bytes)   → broadcast raw binary audio bytes to all clients

    Both are scheduled via asyncio.run_coroutine_threadsafe so they are safe
    to call from any sync thread (TTS thread, pipeline thread, etc.).
    """
    def _sender(msg: dict | None, audio_bytes: bytes | None) -> None:
        if not _loop or not _clients:
            return

        if audio_bytes is not None:
            # Binary frame — send raw bytes
            async def _send_binary():
                dead = set()
                for ws in list(_clients):
                    try:
                        await ws.send(audio_bytes)
                    except Exception:
                        dead.add(ws)
                _clients.difference_update(dead)
            asyncio.run_coroutine_threadsafe(_send_binary(), _loop)
        elif msg is not None:
            # JSON frame
            async def _send_json():
                text = json.dumps(msg)
                dead = set()
                for ws in list(_clients):
                    try:
                        await ws.send(text)
                    except Exception:
                        dead.add(ws)
                _clients.difference_update(dead)
            asyncio.run_coroutine_threadsafe(_send_json(), _loop)

    return _sender


# ──────────────────────────────────────────────────
# VAD oracle callbacks
# ──────────────────────────────────────────────────

def _on_speech_start(start_s: float) -> None:
    """Fired by VADOracle at confirmed onset (pre-buffer already applied)."""
    print(f"  [vad] ▶ speech start  t={start_s:.3f}s  (incl 2.0s pre-buffer)", flush=True)
    # Start a fresh StreamingTranscriber for live word display this turn.
    _active_transcriber[0] = _make_transcriber()
    # Barge-in: if bot is speaking, abort immediately.
    if _current_voice_state == "speaking" and not _interrupt_event.is_set():
        print(f"  [vad] ⚡ BARGE-IN  speech_start={start_s:.3f}s", flush=True)
        abort_speaking()
        _interrupt_event.set()
    # Predictive preload: warm the 3b model while Whisper transcribes.
    # By the time STT finishes (~1-2s), the model cold-load (~2-3s) may already
    # be done — eliminating the delay on the first token of the response.
    if _loop:
        asyncio.run_coroutine_threadsafe(
            _preload_model(LLM_CONFIG.model, keep_alive_s=120),
            _loop,
        )


def _on_speech_end(start_s: float, end_s: float) -> None:
    """Fired by VADOracle at confirmed silence end (post-buffer already applied)."""
    dur = end_s - start_s
    print(f"  [vad] ■ speech end  [{start_s:.3f}s → {end_s:.3f}s]  {dur:.3f}s", flush=True)
    _speech_events.put_nowait((start_s, end_s))


def _on_speech_chunk(chunk: np.ndarray) -> None:
    """Forward each 16 ms speech frame to StreamingTranscriber for live display."""
    t = _active_transcriber[0]
    if t is not None:
        t.feed(chunk)


def _on_first_silence(snapshot: np.ndarray) -> None:
    """Start background Whisper at first silence (early-STT optimisation)."""
    t = _active_transcriber[0]
    if t is not None:
        t.start_finalize(snapshot)


# ──────────────────────────────────────────────────
# Audio ingestion loop — runs continuously in its own thread
# ──────────────────────────────────────────────────

def _audio_ingestion_loop():
    """
    Audio ingestion: write ALL mic audio to rolling buffer (never gated),
    then feed VADOracle in parallel to get speech timestamps.

    Key difference from old _vad_loop:
      OLD: StreamingVAD accumulated audio → returned utterance → _speech_queue.
           Mic was effectively gated; audio only reached STT after VAD onset.
      NEW: _audio_buffer.write() happens UNCONDITIONALLY for every frame.
           VADOracle emits timestamps only; never gates audio.
           _speech_events queue receives (start_s, end_s) pairs.
           _run_turn extracts audio from buffer using those timestamps,
           going BACK IN TIME to include audio before VAD onset.
    """
    global _audio_buffer

    GRACE_S = 1.0   # seconds after "speaking" start to ignore VAD (TTS echo suppression)

    _audio_buffer = RollingAudioBuffer(capacity_s=60, sample_rate=16_000)

    # ── VAD model RAM note (A4) ───────────────────────────────────────────────
    # SileroVAD (~8 MB) is loaded here and stays resident in RAM permanently.
    # This is intentional — VAD must be always-on so we never gate the mic.
    # 8 MB is negligible; do NOT lazy-load or unload between utterances.
    oracle = VADOracle(
        onset_threshold  = 0.50,
        offset_threshold = 0.35,
        onset_count      = 2,
        offset_count     = 45,      # 45 × 16 ms = 720 ms silence → speech_end fires
        pre_buffer_s     = 2.0,
        post_buffer_s    = 2.0,
        on_speech_start  = _on_speech_start,
        on_speech_end    = _on_speech_end,
        on_speech_chunk  = _on_speech_chunk,
        on_first_silence = _on_first_silence,
    )

    print("  [vad] VADOracle ready  (grace=1000ms  offset_count=45×16ms=720ms  onset_count=2  pre_buffer=2.0s  post_buffer=2.0s)", flush=True)

    _prev_state     : str   = ""
    _speaking_since : float = 0.0

    while True:
        try:
            raw = _audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # ── Validate ────────────────────────────────────
        if len(raw) < 8:
            continue
        src_sr = int(np.frombuffer(raw[:4], dtype=np.uint32)[0])
        if src_sr == 0 or src_sr > 192_000:
            continue

        # ── Decode & resample to 16 kHz ──────────────────
        samples = np.frombuffer(raw[4:], dtype=np.float32).copy()
        samples = _resample(samples, src_sr, 16_000)
        np.clip(samples, -1.0, 1.0, out=samples)

        # ── ALWAYS write to rolling buffer — mic is NEVER gated ──────────
        _audio_buffer.write(samples)
        current_time = _audio_buffer.current_time_s

        state = _current_voice_state

        # ── Detect state transitions — reset LSTM to prevent bleed ───────
        if state != _prev_state:
            if state == "speaking":
                oracle.reset()
                _speaking_since = current_time
                print("  [vad] → speaking  (LSTM reset, grace period armed)", flush=True)
            elif state == "listening":
                oracle.reset()
                print(f"  [vad] → listening  (LSTM reset, from {_prev_state})", flush=True)
            _prev_state = state

        # ── Grace period during speaking: buffer receives audio but VAD ──
        # is silenced to prevent TTS echo from triggering false barge-in.
        if state == "speaking" and (current_time - _speaking_since) < GRACE_S:
            continue   # audio already written to buffer above; skip oracle

        # ── Feed oracle — fires timestamp callbacks, never accumulates audio
        oracle.process(samples, current_time)


# ──────────────────────────────────────────────────
# Memory type helpers
# ──────────────────────────────────────────────────

_MEMORY_TYPE_MAP = {
    "PersonalMemory":     "personal",
    "ProjectMemory":      "project",
    "ArchitectureMemory": "project",
    "DecisionMemory":     "decision",
    "ReflectionMemory":   "reflection",
    "ActionMemory":       "action",
    "IdeaMemory":         "idea",
    "ConsolidatedMemory": "idea",
    "PlannerMemory":      "action",   # planner actions surface as action-type memories
}

_MEMORY_IMPORTANCE = {
    "PersonalMemory":     8,
    "ConsolidatedMemory": 9,
    "DecisionMemory":     7,
    "PlannerMemory":      7,   # plan completions are significant events
    "ProjectMemory":      6,
    "ArchitectureMemory": 6,
    "ActionMemory":       5,
    "IdeaMemory":         5,
    "ReflectionMemory":   4,
}


# ──────────────────────────────────────────────────
# Identity extraction
# ──────────────────────────────────────────────────

def _extract_identity_facts(user_text: str) -> list[dict]:
    facts = []
    if m := re.search(r"my name is\s+([A-Za-z]+)", user_text, re.IGNORECASE):
        facts.append({"text": f"User name is {m.group(1)}", "memory_type": "PersonalMemory", "project_reference": "UserProfile"})
    if m := re.search(r"i am\s+(\d{1,3})", user_text, re.IGNORECASE):
        facts.append({"text": f"User age is {m.group(1)}", "memory_type": "PersonalMemory", "project_reference": "UserProfile"})
    if m := re.search(r"my friend'?s name is\s+([A-Za-z]+)", user_text, re.IGNORECASE):
        facts.append({"text": f"User friend's name is {m.group(1)}", "memory_type": "PersonalMemory", "project_reference": "UserProfile"})
    return facts


# ──────────────────────────────────────────────────
# Memory retrieval + context builder
# ──────────────────────────────────────────────────

def _build_memory_context(user_text: str) -> str:
    try:
        memories = retrieve_memories(user_text, top_k=5)
    except Exception as e:
        print(f"    [memory] retrieval failed: {e}")
        return user_text

    for m in memories:
        _emit("MEMORY_EVENT", {
            "retrieved":  True,
            "id":         m["memory_id"],
            "type":       _MEMORY_TYPE_MAP.get(m["memory_type"], "idea"),
            "importance": round(m["score"] * 10, 1),
            "summary":    m.get("summary", ""),
        })

    if not memories:
        print("    [memory] none retrieved")
        return user_text

    type_counts = Counter(m.get("memory_type", "Unknown") for m in memories)
    print(f"    [memory] {len(memories)} retrieved  —  " +
          "  ".join(f"{t}:{n}" for t, n in type_counts.most_common()))

    consolidated, personal, strategic, reflections = [], [], [], []
    # Identity de-duplication: track the highest-scored entry for each identity key.
    # This prevents conflicting names from different sessions appearing simultaneously.
    _identity_best: dict[str, tuple[float, str]] = {}  # key → (score, summary)

    for m in memories:
        summary = m.get("summary", "")
        mt      = m.get("memory_type", "")
        score   = m.get("score", 0.0)

        # Check for identity facts (User name, User age, etc.)
        summary_lower = summary.lower()
        identity_key = None
        if summary_lower.startswith("user name"):
            identity_key = "user_name"
        elif summary_lower.startswith("user age"):
            identity_key = "user_age"
        elif summary_lower.startswith("user friend"):
            identity_key = "user_friend"

        if identity_key is not None:
            # Keep only the highest-scored (most recent + relevant) identity entry
            prev_score, _ = _identity_best.get(identity_key, (-1.0, ""))
            if score > prev_score:
                _identity_best[identity_key] = (score, summary)
            continue  # handled separately — don't add to personal list yet

        if mt == "ConsolidatedMemory":
            consolidated.append(summary)
        elif mt in ("ProjectMemory", "DecisionMemory", "ArchitectureMemory", "ActionMemory"):
            strategic.append(summary)
        elif mt == "ReflectionMemory":
            reflections.append(summary)

    # Inject exactly one winner per identity key (no conflicting names)
    personal = [summary for _, summary in _identity_best.values()]

    ordered = consolidated[:2] + personal[:3] + strategic[:2] + reflections[:1]
    if not ordered:
        return user_text

    context = "Relevant long-term memory:\n" + "\n".join(f"- {l}" for l in ordered)
    return f"{context}\n\nUser: {user_text}"


# ──────────────────────────────────────────────────
# Token broadcaster
# ──────────────────────────────────────────────────

def _token_broadcaster(token_gen, interrupt_event=None):
    """
    Wrap an LLM generator: emit LLM_TOKEN events while yielding to TTS.

    Stops emitting as soon as interrupt_event is set so the frontend doesn't
    keep receiving tokens after a barge-in.  The generator itself keeps running
    in its producer thread (inside speak_stream) until it's naturally exhausted.
    """
    _token_count = 0
    for token in token_gen:
        if interrupt_event and interrupt_event.is_set():
            _emit("LLM_TOKEN", {"token": "", "done": True})
            return
        _emit("LLM_TOKEN", {"token": token, "done": False})
        _token_count += 1
        yield token
    _emit("LLM_TOKEN", {"token": "", "done": True, "token_count": _token_count})


# ──────────────────────────────────────────────────
# Control-tag stripping (safety net before TTS)
# ──────────────────────────────────────────────────

def strip_control_tags(text: str) -> str:
    """
    Remove <start_plan>…</start_plan> and <tool_call>…</tool_call> blocks
    from any text string before it reaches TTS synthesis.

    These tags should never appear in spoken output.  The system prompt and
    summarize-specific suffix prevent them in the first place; this is the
    last-resort safety net.
    """
    text = re.sub(r"<start_plan>.*?</start_plan>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool_call>.*?</tool_call>",   "", text, flags=re.DOTALL)
    return text.strip()


# System-prompt suffix injected into every plugin-summarize LLM call.
# Prevents the LLM from emitting <start_plan> or <tool_call> tags when it
# only needs to produce a 1-3 sentence spoken summary.
_PLUGIN_SUMMARIZE_SUFFIX = (
    "CRITICAL: You are summarizing a data result for the user.  "
    "Do NOT emit <start_plan> tags.  Do NOT emit <tool_call> tags.  "
    "Do NOT use markdown, bullet points, or lists.  "
    "Respond in plain spoken English only — 1 to 3 sentences maximum."
)


# ──────────────────────────────────────────────────
# Plugin helpers
# ──────────────────────────────────────────────────

def _handle_plugin_result(result: dict, tracker: LatencyTracker) -> str:
    _emit("PLUGIN_ACTION", {
        "plugin": result["plugin"],
        "action": result["action"],
        "result": result["text"][:200],
        "direct": result["direct"],
    })
    if result["direct"]:
        _emit("VOICE_STATE", {"state": "speaking"})
        _interrupt_event.clear()
        if _tts_enabled:
            with tracker.step("TTS (plugin direct)"):
                try:
                    speak(result["text"], _interrupt_event)
                except Exception as e:
                    print(f"    [tts] plugin speak error: {e}")
        else:
            print("  [tts] disabled by user — skipping synthesis", flush=True)
        print(f"\n  Plugin: {result['text']}\n")
        return result["text"]
    else:
        summary_prompt = (
            "You are Small O. Summarize the following data in 1-2 friendly spoken sentences. "
            "Be concise and natural. Highlight the most useful information.\n\n"
            f"Data:\n{result['text']}"
        )
        _emit("VOICE_STATE", {"state": "speaking"})
        _interrupt_event.clear()
        with tracker.step("LLM + TTS (plugin summarize)"):
            try:
                if _tts_enabled:
                    ai_text, tts_timing = speak_stream(
                        _token_broadcaster(
                            ask_llm_stream(summary_prompt, system_suffix=_PLUGIN_SUMMARIZE_SUFFIX),
                            _interrupt_event,
                        ),
                        _interrupt_event,
                    )
                else:
                    print("  [tts] disabled by user — skipping synthesis", flush=True)
                    ai_text = "".join(_token_broadcaster(
                        ask_llm_stream(summary_prompt, system_suffix=_PLUGIN_SUMMARIZE_SUFFIX),
                        _interrupt_event,
                    ))
                    tts_timing = {"first_word_secs": 0.0, "total_secs": 0.0}
            except Exception as e:
                print(f"    [tts] plugin speak_stream error: {e}")
                return ""
        # Safety net: strip any control tags that escaped the system-prompt guard
        if "<start_plan>" in ai_text or "<tool_call>" in ai_text:
            print("  [plugin] ⚠ stripping control tags that leaked into plugin summary", flush=True)
            ai_text = strip_control_tags(ai_text)
        if _tts_enabled:
            print(f"    [tts] first word: {tts_timing['first_word_secs']:.3f}s  |  total: {tts_timing['total_secs']:.3f}s")
        print(f"\n  Plugin summary: {ai_text}\n")
        return ai_text


def _store_action_memory(user_text: str, spoken_text: str, result: dict):
    def _run():
        try:
            memory_id = insert_memory({
                "text": (
                    f"User requested: {user_text}\n"
                    f"Plugin: {result['plugin']} / Action: {result['action']}\n"
                    f"Response: {spoken_text}"
                ),
                "memory_type": "ActionMemory",
                "project_reference": f"Plugin:{result['plugin']}",
                "source": "plugin",
            })
            if memory_id:
                _emit("MEMORY_EVENT", {
                    "retrieved": False,
                    "id":        memory_id,
                    "type":      "action",
                    "importance": _MEMORY_IMPORTANCE["ActionMemory"],
                    "summary":   f"{result['plugin']}: {spoken_text[:100]}",
                })
        except Exception as e:
            print(f"    [memory] action memory insert failed: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ──────────────────────────────────────────────────
# System stats loop
# ──────────────────────────────────────────────────

def _stats_loop():
    psutil.cpu_percent(interval=None)   # prime baseline (first call always returns 0.0)
    time.sleep(0.5)
    while True:
        try:
            battery = psutil.sensors_battery()
            _emit("SYSTEM_STATS", {
                "cpu":     psutil.cpu_percent(interval=None),
                "ram":     psutil.virtual_memory().percent,
                "battery": round(battery.percent) if battery else 100,
            })
        except Exception:
            pass
        time.sleep(2)


# ──────────────────────────────────────────────────
# Main pipeline loop
# ──────────────────────────────────────────────────

def _pipeline_loop():
    print("  Warming up models...", flush=True)
    t0 = time.perf_counter(); stt_warmup(); print(f"    STT  ready  ({time.perf_counter()-t0:.2f}s)", flush=True)
    t0 = time.perf_counter(); tts_warmup(); print(f"    TTS  ready  ({time.perf_counter()-t0:.2f}s)", flush=True)
    t0 = time.perf_counter(); llm_warmup(); print(f"    LLM  ready  ({time.perf_counter()-t0:.2f}s)", flush=True)

    # ── RAM report after models load ─────────────────────────────────────────
    _ram_available = get_available_ram_gb()
    _ram_pressure  = get_memory_pressure()
    print(f"  [ram] {_ram_available:.1f} GB available — pressure: {_ram_pressure}", flush=True)
    if _ram_pressure == "high":
        print(
            "  [ram] ⚠ WARNING: very low RAM — 7b planner disabled, close other apps",
            flush=True,
        )
    elif _ram_pressure == "medium":
        print(
            "  [ram] ⚡ CAUTION: limited RAM — 7b will only load if > 3 GB free at plan time",
            flush=True,
        )
    else:
        print(
            f"  [ram] ✓ OK — qwen2.5:7b planner {'enabled' if can_load_7b() else 'unavailable'}",
            flush=True,
        )
    _emit("SYSTEM_EVENT", {
        "event":        "ram_report",
        "message":      f"{_ram_available:.1f} GB free — {_ram_pressure} pressure",
        "available_gb": round(_ram_available, 2),
    })

    # ── Predictive preload: warm 3b model after startup so first turn is fast ──
    if _loop:
        asyncio.run_coroutine_threadsafe(
            _preload_model(LLM_CONFIG.model, keep_alive_s=120),
            _loop,
        )

    print("\n  Loading plugins...")
    try:
        router = PluginRouter()
    except Exception as e:
        print(f"  [pipeline] plugin router failed to load: {e} — continuing without plugins")
        router = None
    print()

    turn               = 0
    came_from_barge_in = False
    while True:
        turn += 1
        tracker = LatencyTracker(turn=turn)
        print(f"\n{'─'*54}")
        print(f"  Turn {turn}")
        print(f"{'─'*54}")

        # ── Top-level crash guard — one bad turn never kills the loop ──────
        global _turn_in_progress, _turn_started_at
        try:
            _turn_in_progress = True
            _turn_started_at  = time.time()
            came_from_barge_in = _run_turn(turn, tracker, router, came_from_barge_in)
        except Exception as e:
            came_from_barge_in = False
            print(f"\n  [pipeline] !! UNHANDLED EXCEPTION in turn {turn}: {e}")
            import traceback; traceback.print_exc()
            _emit("VOICE_STATE", {"state": "idle"})
            time.sleep(1)   # brief pause before retrying
        finally:
            _turn_in_progress = False


def _run_turn(turn: int, tracker: LatencyTracker, router,
              came_from_barge_in: bool = False) -> bool:
    """
    Execute one full pipeline turn.

    Returns True if this turn ended with a barge-in (caller should NOT clear
    the speech events queue at the start of the next turn), False otherwise.
    Called inside the crash-guard try/except in _pipeline_loop.
    """
    global _interrupted_partial

    # ── Clear stale speech events from previous turn ─────────────────────
    # Skip clearing if we came from a barge-in: the barge-in speech_end event
    # may still be pending in _speech_events and we must not discard it.
    if not came_from_barge_in:
        cleared = 0
        while not _speech_events.empty():
            try:
                _speech_events.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            print(f"  [pipeline] cleared {cleared} stale speech event(s)")

    # ── Wait for input: text OR speech (whichever arrives first) ─────────────
    _emit("VOICE_STATE", {"state": "listening"})
    print(f"  [pipeline] VOICE_STATE → listening")

    # Variables set by whichever input branch fires
    user_text:  str   = ""
    rec_secs:   float = 0.0
    trans_secs: float = 0.0
    _text_input_branch = False

    # ── Conversation-active idle check ────────────────────────────────────────
    # If the user has been silent for > 90 s, mark conversation as inactive so
    # the next LLM call uses KEEP_ALIVE_IDLE (0 s) and frees RAM when idle.
    global _last_turn_time
    if _last_turn_time > 0 and (time.time() - _last_turn_time) > _CONVERSATION_IDLE_S:
        set_conversation_active(False)
        print("  [pipeline] conversation idle >90 s — model keep_alive → IDLE", flush=True)

    # ── Fast path: check for immediately-available text input ─────────────────
    # Text input turns bypass the STT step entirely (source="text").
    # This check is non-blocking; if no text is ready we fall into the voice
    # path where STT timing is properly tracked.
    try:
        user_text = _text_input_queue.get_nowait()
        _text_input_branch = True
        tracker.header("source: text input")
        print(f"  [text_input] processing: {user_text[:80]!r}", flush=True)
    except queue.Empty:
        # ── Voice path — track STT timing ─────────────────────────────────────
        with tracker.step("STT"):
            rec_start = time.perf_counter()

            # Still poll text queue inside the loop so text arriving AFTER we
            # entered the voice wait is picked up within 500 ms.
            TIMEOUT_S = 30.0
            while True:
                try:
                    user_text = _text_input_queue.get_nowait()
                    _text_input_branch = True
                    # Cancel the STT step so it doesn't pollute the turn summary
                    # with misleading "STT Xs" timing (that was just idle waiting).
                    tracker.cancel_current_step()
                    tracker.header("source: text input")
                    print(f"  [text_input] processing: {user_text[:80]!r}", flush=True)
                    break
                except queue.Empty:
                    pass

                # ── Speech event (normal voice path) ───────────────────
                try:
                    speech_start_s, speech_end_s = _speech_events.get(timeout=0.5)
                    audio_data = _audio_buffer.read_window(speech_start_s, speech_end_s)

                    # Guard: VAD can fire on a window with no buffered frames.
                    if audio_data is None or len(audio_data) == 0:
                        print(
                            f"  [stt] skipping transcription — empty audio window "
                            f"[{speech_start_s:.3f}s → {speech_end_s:.3f}s]",
                            flush=True,
                        )
                        _emit("VOICE_STATE", {"state": "idle"})
                        return False

                    dur      = len(audio_data) / 16_000
                    rec_secs = time.perf_counter() - rec_start

                    print(
                        f"  [stt] window [{speech_start_s:.3f}s → {speech_end_s:.3f}s]"
                        f"  {dur:.2f}s audio  (waited {rec_secs:.3f}s)",
                        flush=True,
                    )
                    tracker.note(f"utterance: {dur:.2f}s  [{speech_start_s:.3f}s → {speech_end_s:.3f}s]")

                    transcriber             = _active_transcriber[0]
                    _active_transcriber[0] = _make_transcriber()

                    try:
                        if transcriber is not None:
                            user_text, trans_secs = transcriber.finalize(audio_data)
                            print(f"  [stt] ✓ streaming finalize  ({trans_secs:.3f}s Whisper)", flush=True)
                        else:
                            user_text, trans_secs = transcribe(audio_data)
                    except Exception as e:
                        print(f"  [stt] transcribe failed: {e}")
                        _emit("VOICE_STATE", {"state": "idle"})
                        return False

                    tracker.note(f"transcript: '{user_text[:70]}{'…' if len(user_text)>70 else ''}'  ({trans_secs:.3f}s)")
                    break
                except queue.Empty:
                    pass

                # ── Timeout guard ──────────────────────────────────────
                if time.perf_counter() - rec_start >= TIMEOUT_S:
                    print("  [pipeline] no input for 30 s — going idle")
                    _emit("VOICE_STATE", {"state": "idle"})
                    return False  # caller will check 90 s idle on next turn

        if not _text_input_branch:
            print(f"  [stt] wait={rec_secs:.3f}s  transcription={trans_secs:.3f}s")

    print(f"\n  {'You typed' if _text_input_branch else 'You said'}: {user_text}\n")

    # ── Mark conversation as active — model stays warm for next 120 s ─────────
    _last_turn_time = time.time()
    set_conversation_active(True)
    # Preload the 3b model now so it's warm before we need it this turn.
    # Fire-and-forget; LLM call will race against this and usually win.
    if _loop:
        asyncio.run_coroutine_threadsafe(
            _preload_model(LLM_CONFIG.model, keep_alive_s=120),
            _loop,
        )

    if not user_text.strip():
        _emit("VOICE_STATE", {"state": "idle"})
        return False

    if user_text.lower().strip() in ("exit", "quit"):
        _emit("VOICE_STATE", {"state": "idle"})
        return False

    _emit("STT_RESULT", {
        "text":               user_text,
        "recording_time":     round(rec_secs, 3),
        "transcription_time": round(trans_secs, 3),
    })

    # ── Plan cancellation (fast path — BEFORE thinking state, no LLM call) ─
    # Handles both "stop" with an active plan and "stop" with nothing running.
    # Checked before plugin routing and before any LLM call.
    _user_lower = user_text.lower().strip()
    _is_cancel = (
        any(_user_lower == w or _user_lower.startswith(w + " ") for w in _CANCEL_WORDS)
        or any(phrase in _user_lower for phrase in ("stop the", "cancel the", "abort the"))
    )
    if _is_cancel:
        if _current_plan_task and not _current_plan_task.done():
            asyncio.run_coroutine_threadsafe(_cancel_plan(), _loop)
            _emit("VOICE_STATE", {"state": "speaking"})
            _interrupt_event.clear()
            try:
                speak("Stopped.", _interrupt_event)
            except Exception:
                pass
        else:
            _emit("VOICE_STATE", {"state": "speaking"})
            _interrupt_event.clear()
            try:
                speak("Nothing is running.", _interrupt_event)
            except Exception:
                pass
        _emit("VOICE_STATE", {"state": "idle"})
        tracker.summary()
        return False

    _emit("VOICE_STATE", {"state": "thinking"})

    # ── Plugin routing ───────────────────────────────────────────────────
    plugin_result = None
    if router is not None:
        try:
            with tracker.step("Plugin Router"):
                plugin_result = router.route(user_text)
        except Exception as e:
            print(f"  [plugin] router error: {e}")

    if plugin_result is not None:
        try:
            spoken = _handle_plugin_result(plugin_result, tracker)
            _store_action_memory(user_text, spoken, plugin_result)
        except Exception as e:
            print(f"  [plugin] handle result error: {e}")
        _emit("VOICE_STATE", {"state": "idle"})
        tracker.summary()
        return False

    # ── Identity extraction ──────────────────────────────────────────────
    try:
        with tracker.step("Identity Extraction"):
            identity_facts = _extract_identity_facts(user_text)
    except Exception as e:
        print(f"  [identity] extraction error: {e}")
        identity_facts = []

    try:
        with tracker.step("Identity Memory Insert"):
            for fact in identity_facts:
                memory_id = insert_memory(fact)
                if memory_id:
                    _emit("MEMORY_EVENT", {
                        "retrieved":  False,
                        "id":         memory_id,
                        "type":       "personal",
                        "importance": _MEMORY_IMPORTANCE["PersonalMemory"],
                        "summary":    fact["text"],
                    })
    except Exception as e:
        print(f"  [identity] memory insert error: {e}")

    # ── Memory retrieval ─────────────────────────────────────────────────
    try:
        with tracker.step("Memory Retrieval"):
            prompt = _build_memory_context(user_text)
    except Exception as e:
        print(f"  [memory] context build error: {e}")
        prompt = user_text

    # ── Interrupted-response context ─────────────────────────────────────
    # If the bot was cut off mid-response, prepend what it had already said
    # so it can acknowledge the interruption and seamlessly respond to the
    # new query (or pick up where it left off if the user asks it to).
    if _interrupted_partial:
        snip = _interrupted_partial[:400] + ("…" if len(_interrupted_partial) > 400 else "")
        prompt = (
            f"[Context: you were mid-response saying \"{snip}\" when the user interrupted you. "
            f"You don't need to repeat it; just respond naturally to what they said next.]\n\n"
            f"{prompt}"
        )
        print(f"  [pipeline] injecting interrupted context ({len(_interrupted_partial)} chars)")
        _interrupted_partial = ""

    # ── LLM (with plan-trigger and tool detection) ───────────────────────
    # Inject user context + recent planner actions into every LLM call.
    user_ctx_suffix = get_context_prompt()
    recent_actions_ctx = get_recent_actions_prompt()
    if recent_actions_ctx:
        user_ctx_suffix = (user_ctx_suffix + "\n\n" + recent_actions_ctx).strip()

    # ── Inject last plan result (one-shot, consumed once per turn) ────────
    # If the planner completed within the last 300 s, surface what it did so
    # the conversational LLM can answer follow-up questions like "where did
    # you save that?" without needing a memory retrieval round-trip.
    try:
        import planner.planner as _planner_mod
        import time as _time
        plan_result = _planner_mod.get_last_plan_result()
        if plan_result and (_time.time() - plan_result["completed_at"]) < 300:
            tool_names = ", ".join(t["tool"] for t in plan_result["tool_calls"])
            snippets = "; ".join(
                t["result_snippet"] for t in plan_result["tool_calls"][:3] if t["result_snippet"]
            )
            plan_context = (
                f"\n[Recent action I just completed]\n"
                f"Goal: {plan_result['goal']}\n"
                f"What I did: {plan_result['summary']}\n"
                f"Tools used: {tool_names}\n"
                f"Key results: {snippets}"
            )
            user_ctx_suffix = (user_ctx_suffix + plan_context).strip()
            _planner_mod.clear_last_plan_result()
    except Exception as _plan_ctx_exc:
        pass  # non-fatal — plan context is best-effort

    print(f"  [pipeline] calling ask_llm_turn  |  prompt {len(prompt):,} chars", flush=True)
    try:
        llm_result = ask_llm_turn(prompt, system_suffix=user_ctx_suffix)
    except Exception as e:
        print(f"  [llm] ask_llm_turn failed: {e}")
        _emit("VOICE_STATE", {"state": "idle"})
        tracker.summary()
        return False

    # ── Plan trigger detected — hand off to autonomous planner ───────────
    if isinstance(llm_result, dict) and llm_result.get("type") == "plan_trigger":
        goal = llm_result["goal"]
        print(f"  [pipeline] 🗺 plan trigger → '{goal}'", flush=True)
        # Drain any stale signal from a previous plan
        while not _plan_result_queue.empty():
            try:
                _plan_result_queue.get_nowait()
            except queue.Empty:
                break
        # Launch plan on main asyncio loop (non-blocking from pipeline thread)
        asyncio.run_coroutine_threadsafe(_launch_plan(goal), _loop)
        # Wait for planner to signal: "MULTI" → multi-step (say "On it"),
        # or a result string → single-tool completed (generate natural response).
        # Hard 45s deadline — no turn ever hangs forever (FIX1A).
        # Single-tool tools should complete well within this window.
        plan_signal = None
        deadline = time.time() + 45  # was 120 — hard cap prevents indefinite hangs
        while time.time() < deadline:
            try:
                plan_signal = _plan_result_queue.get(timeout=0.25)
                break
            except queue.Empty:
                if not _text_input_queue.empty() or _interrupt_event.is_set():
                    break
        _emit("VOICE_STATE", {"state": "speaking"})
        _interrupt_event.clear()
        if plan_signal is None or plan_signal in ("MULTI", "TIMEOUT"):
            # Multi-step plan, 45s timeout, or watchdog reset — speak generic ack.
            # The plan is either still running (MULTI) or timed out (TIMEOUT/None).
            # In the MULTI case the planner will speak its own completion summary.
            try:
                speak("On it. I'll let you know when it's done.", _interrupt_event)
            except Exception:
                pass
        else:
            # Single-tool completed — ask LLM to produce a natural spoken response
            # from the raw tool result instead of speaking the JSON/plain text directly.
            # This is FIX1C: ensures something always plays even if result is terse.
            print(f"  [pipeline] single-tool result → generating natural response", flush=True)
            tool_suffix = (user_ctx_suffix + f"\n[Tool result]: {plan_signal}").strip()
            try:
                llm_resp = ask_llm_turn(prompt, system_suffix=tool_suffix)
                if isinstance(llm_resp, dict):
                    # plan_trigger inside plan response — shouldn't happen, speak raw
                    speak(str(plan_signal)[:300], _interrupt_event)
                elif _tts_enabled:
                    speak_stream(
                        _token_broadcaster(llm_resp, _interrupt_event),
                        _interrupt_event,
                    )
                else:
                    print("  [tts] disabled — skipping synthesis", flush=True)
                    "".join(_token_broadcaster(llm_resp, _interrupt_event))
            except Exception as _resp_err:
                print(f"  [pipeline] LLM response for tool result failed: {_resp_err} — speaking raw", flush=True)
                try:
                    speak(str(plan_signal)[:300], _interrupt_event)
                except Exception:
                    pass
        _emit("VOICE_STATE", {"state": "idle"})
        tracker.summary()
        return False

    # ── Normal streaming response ────────────────────────────────────────
    print(f"  [pipeline] VOICE_STATE → speaking", flush=True)
    _emit("VOICE_STATE", {"state": "speaking"})
    _interrupt_event.clear()   # arm: VAD can now set this on barge-in

    try:
        if not _tts_enabled:
            # TTS disabled by user — drain LLM generator to emit LLM_TOKEN events
            # to the frontend (text still streams), but skip all audio synthesis.
            print("  [tts] disabled by user — skipping synthesis", flush=True)
            t0       = time.perf_counter()
            ai_text  = "".join(_token_broadcaster(llm_result, _interrupt_event))
            llm_secs = time.perf_counter() - t0
            tracker.record("LLM", llm_secs, [
                f"text-only mode — {len(ai_text)} chars in {llm_secs:.3f}s"
            ])
        else:
            ai_text, tts_timing = speak_stream(
                _token_broadcaster(llm_result, _interrupt_event),
                _interrupt_event,
            )
            first_token = tts_timing.get("first_token_secs", tts_timing["first_word_secs"])
            first_audio = tts_timing["first_word_secs"]
            total_t     = tts_timing["total_secs"]
            n_tok       = tts_timing.get("token_count", "?")
            n_sent      = tts_timing.get("sentence_count", "?")
            tts_synth_s = max(0.0, first_audio - first_token)
            speaking_s  = max(0.0, total_t - first_audio)

            tracker.record("LLM", first_token, [
                f"first token in {first_token:.3f}s  ({n_tok} tokens total)"
            ])
            tracker.record("TTS", tts_synth_s, [
                f"Kokoro synth first sentence: {tts_synth_s:.3f}s  ({n_sent} sentences)"
            ])
            tracker.record("Speaking", speaking_s, [
                f"audio playback: {speaking_s:.3f}s"
            ])
        print(f"\n  AI: {ai_text}\n")
    except Exception as e:
        print(f"  [llm/tts] error: {e}")
        _emit("VOICE_STATE", {"state": "idle"})
        tracker.summary()
        return

    # ── Barge-in: save partial response, decide next state ───────────────
    if _interrupt_event.is_set():
        if ai_text.strip():
            # Bot had started speaking — real barge-in.  Save partial so the next
            # turn can acknowledge it, then listen for the user's follow-up.
            with _barge_in_lock:
                _interrupted_partial = ai_text.strip()
            chars = len(_interrupted_partial)
            snip  = _interrupted_partial[:80] + ("…" if chars > 80 else "")
            print(f"  [pipeline] ⚡ BARGE-IN — partial saved ({chars} chars): \"{snip}\"")
            print("  [pipeline] restarting turn from listening state")
            _emit("VOICE_STATE", {"state": "listening"})
            tracker.summary()
            return True   # barge-in: next turn should NOT clear speech events
        else:
            # Bot hadn't spoken yet — could be a very quick interrupt or false
            # barge-in.  If a speech_end event is already queued, process it;
            # otherwise go idle.
            if not _speech_events.empty():
                print("  [pipeline] ⚡ BARGE-IN — bot silent, speech event pending — restarting")
                _emit("VOICE_STATE", {"state": "listening"})
                tracker.summary()
                return True
            print("  [pipeline] ⚡ False barge-in (no text, no event) — going idle")
            _emit("VOICE_STATE", {"state": "idle"})
            tracker.summary()
            return False

    _emit("VOICE_STATE", {"state": "idle"})
    tracker.summary()

    # ── Reflection memory (background) ───────────────────────────────────
    def _store_reflection(ut=user_text, at=ai_text):
        try:
            memory_id = insert_memory({
                "text":              f"User: {ut}\nAssistant: {at}",
                "memory_type":       "ReflectionMemory",
                "project_reference": "VoiceInteraction",
            })
            if memory_id:
                _emit("MEMORY_EVENT", {
                    "retrieved":  False,
                    "id":         memory_id,
                    "type":       "reflection",
                    "importance": _MEMORY_IMPORTANCE["ReflectionMemory"],
                    "summary":    f"You: {ut[:50]}… / AI: {at[:50]}…",
                })
        except Exception as e:
            print(f"  [memory] reflection insert failed: {e}")

    threading.Thread(target=_store_reflection, daemon=True).start()
    return False


# ──────────────────────────────────────────────────
# WebSocket server
# ──────────────────────────────────────────────────

async def _ws_handler(ws):
    _clients.add(ws)
    print(f"  [ws] client connected  ({len(_clients)} total)  addr={ws.remote_address}")

    # Send current voice state immediately so late-connecting frontends sync up
    try:
        await ws.send(json.dumps({"event": "VOICE_STATE", "data": {"state": _current_voice_state}}))
    except Exception:
        pass

    # Preload 3b model the moment a frontend connects — by the time the user
    # types their first message the model is already warm (120 s window).
    asyncio.ensure_future(_preload_model(LLM_CONFIG.model, keep_alive_s=120))

    try:
        async for message in ws:
            if isinstance(message, bytes):
                if len(message) < 8:
                    continue
                src_sr = int(np.frombuffer(message[:4], dtype=np.uint32)[0])
                if src_sr == 0 or src_sr > 192_000:
                    continue
                _audio_queue.put_nowait(message)
            else:
                try:
                    ctrl = json.loads(message)
                    ev   = ctrl.get("event", "")

                    if ev == "ping":
                        await ws.send(json.dumps({"event": "pong", "data": {}}))

                    elif ev == "UPDATE_USER_CONTEXT":
                        # Frontend (or future sub-agent) can update the user model.
                        # e.g. {"event": "UPDATE_USER_CONTEXT", "data": {"key": "name", "value": "Alice"}}
                        key   = ctrl.get("data", {}).get("key")
                        value = ctrl.get("data", {}).get("value")
                        if key:
                            update_user_context(key, value)
                            print(f"  [ws] user_context updated: {key}={value!r}")

                    elif ev == "SET_TTS_ENABLED":
                        global _tts_enabled
                        _tts_enabled = bool(ctrl.get("enabled", True))
                        print(f"  [ws] SET_TTS_ENABLED → {_tts_enabled}", flush=True)

                    elif ev == "TEXT_INPUT":
                        # User typed a message — inject directly into pipeline.
                        text = ctrl.get("data", {}).get("text", "").strip()
                        if text:
                            _text_input_queue.put_nowait(text)
                            print(f"  [ws] TEXT_INPUT: {text[:80]!r}", flush=True)

                    elif ev == "GET_SESSION_FILES":
                        from bot_docs.store import get_session_files
                        from dataclasses import asdict
                        files = get_session_files(SESSION_ID)
                        await ws.send(json.dumps({
                            "event": "FILE_LIST",
                            "data": {"files": [asdict(f) for f in files]},
                        }))

                    elif ev == "DOWNLOAD_FILE":
                        uid = ctrl.get("data", {}).get("uid", "")
                        if uid:
                            from bot_docs.store import get_file_content, get_entry_by_uid
                            from dataclasses import asdict
                            content = get_file_content(uid)
                            entry   = get_entry_by_uid(uid)
                            if content is not None and entry:
                                await ws.send(json.dumps({
                                    "event": "FILE_CONTENT",
                                    "data": {
                                        "uid":       uid,
                                        "title":     entry.title,
                                        "filename":  entry.filename,
                                        "content":   content,
                                        "extension": entry.extension,
                                    },
                                }))

                except Exception:
                    pass
    except Exception:
        pass
    finally:
        _clients.discard(ws)
        print(f"  [ws] client disconnected  ({len(_clients)} remaining)")


async def _main():
    global _loop
    _loop = asyncio.get_running_loop()

    # ── Share the event loop with tool dispatch (reminder asyncio tasks) ──────
    backend_loop_ref.loop = _loop

    # ── Session-ID memory migration (FIX2A — BUG-005) ────────────────────────
    # Adds session_id column to memories if absent and marks pre-existing rows
    # as 'legacy' so cross-session identity contamination can be penalised in
    # retrieval scoring.  Must run before any memory insert or retrieval.
    try:
        from memory_system.db.migrate_session import migrate_add_session_id  # noqa: PLC0415
        migrate_add_session_id()
    except Exception as _mig_exc:
        print(f"  [memory] session migration failed (non-fatal): {_mig_exc}", flush=True)

    # Publish current session_id into the shared ref so insert_pipeline and
    # search.py can tag / penalise memories without a circular import.
    backend_loop_ref.session_id = SESSION_ID
    print(f"  [memory] session context: {SESSION_ID}", flush=True)

    # ── Load persistent user context from disk ───────────────────────────────
    load_user_context()

    # ── bot-docs managed file store ───────────────────────────────────────────
    from bot_docs.store import ensure_dirs, BOT_DOCS_DIR
    ensure_dirs()
    print(f"  [bot-docs] directory ready: {BOT_DOCS_DIR}", flush=True)
    print(f"  [session] ID: {SESSION_ID}", flush=True)
    from tools.file_tool import (
        set_broadcast_fn as _set_file_broadcast,
        set_session_id  as _set_file_session,
    )
    _set_file_broadcast(_emit)
    _set_file_session(SESSION_ID)

    # Register the WS audio sender so TTS can stream audio to browser clients.
    # Must happen after _loop is set (sender uses run_coroutine_threadsafe).
    register_ws_audio_sender(_make_audio_sender())

    # ── Wire reminder tool's broadcast function ───────────────────────────────
    _set_reminder_broadcast(_emit)

    # ── Wire web agent broadcast + warm-up browser ────────────────────────────
    _set_web_broadcast(_emit)
    web_monitor.set_broadcast_fn(_emit)
    try:
        await BrowserManager.get()           # launch Chromium early; shows window now
        print("  [web_agent] Chromium ready", flush=True)
    except Exception as _exc:
        print(f"  [web_agent] Chromium launch failed (will retry on first use): {_exc}", flush=True)

    # ── Wire adapter broadcast functions ──────────────────────────────────────
    _set_research_broadcast(_emit)
    _set_browser_use_broadcast(_emit)
    print("  [adapters] research + browser-use broadcast wired", flush=True)

    # Start background webpage-monitor loop
    web_monitor.run_forever(_loop)
    print("  [monitor] background loop scheduled", flush=True)

    # ── Text-input watcher (file + WS both feed _text_input_queue) ─────────────
    def _on_text_transcript(text: str) -> None:
        """Enqueue typed text exactly like a finished STT result."""
        _text_input_queue.put_nowait(text)

    asyncio.create_task(watch_text_input(_on_text_transcript))

    # ── Memory cap: startup eviction check ───────────────────────────────────
    try:
        from memory_system.embeddings.eviction import (  # noqa: PLC0415
            MAX_MEMORIES, get_memory_count, evict_and_rebuild,
        )
        _mem_count = get_memory_count()
        print(f"  [memory] startup count: {_mem_count}  cap: {MAX_MEMORIES}", flush=True)
        if _mem_count > MAX_MEMORIES:
            print(
                f"  [memory] startup eviction: {_mem_count} > {MAX_MEMORIES} — rebuilding...",
                flush=True,
            )
            await asyncio.get_running_loop().run_in_executor(None, evict_and_rebuild)
    except Exception as _exc:
        print(f"  [memory] startup eviction check failed (non-fatal): {_exc}", flush=True)

    # ── Memory lifecycle background tasks ─────────────────────────────────────
    # run_lifecycle_maintenance: archives stale low-importance memories every 6 h.
    # run_weekly_consolidation:  clusters similar memories (LLM call) every 24 h.
    # Both are synchronous; offloaded to thread pool via run_in_executor.
    # LLM calls inside consolidation use KEEP_ALIVE_IDLE (conversation_active=False
    # since we're in a background thread, not a user turn).
    from memory_system.lifecycle.worker import run_lifecycle_maintenance as _run_maint  # noqa: PLC0415
    from memory_system.lifecycle.consolidator import run_weekly_consolidation as _run_consol  # noqa: PLC0415

    async def _memory_maintenance_loop():
        """Archive stale memories every 6 hours (runs immediately on first tick)."""
        while True:
            try:
                await asyncio.get_running_loop().run_in_executor(None, _run_maint)
                print("  [memory] lifecycle maintenance complete", flush=True)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                print(f"  [memory] lifecycle maintenance error (non-fatal): {exc}", flush=True)
            await asyncio.sleep(6 * 3600)   # every 6 hours

    async def _memory_consolidation_loop():
        """Consolidate similar memory clusters every 24 hours."""
        # First run after 24 h — consolidation is expensive (LLM calls).
        # Do NOT run at startup; wait for data to accumulate.
        await asyncio.sleep(24 * 3600)
        while True:
            try:
                await asyncio.get_running_loop().run_in_executor(None, _run_consol)
                print("  [memory] consolidation complete", flush=True)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                print(f"  [memory] consolidation error (non-fatal): {exc}", flush=True)
            await asyncio.sleep(24 * 3600)   # every 24 hours

    asyncio.create_task(_memory_maintenance_loop(), name="memory-maintenance")
    asyncio.create_task(_memory_consolidation_loop(), name="memory-consolidation")
    print("  [memory] lifecycle tasks scheduled (maintenance=6h, consolidation=24h)", flush=True)

    # ── Pipeline watchdog (FIX1A / Part C) ──────────────────────────────────
    # Detects turns stuck >90 s and resets VOICE_STATE to idle.
    # 45s plan deadline prevents most hangs; this is the last-resort safety net.
    async def _pipeline_watchdog():
        while True:
            await asyncio.sleep(60)
            if _turn_in_progress and time.time() - _turn_started_at > 90:
                print(
                    "[watchdog] turn stuck >90s — resetting pipeline state",
                    flush=True,
                )
                # Can't preempt the thread, but reset voice state so the UI
                # doesn't stay frozen in "thinking" indefinitely.
                _emit("VOICE_STATE", {"state": "idle"})
                # Unblock the pipeline if it's waiting on the plan queue.
                try:
                    _plan_result_queue.put_nowait("TIMEOUT")
                except Exception:
                    pass

    asyncio.create_task(_pipeline_watchdog(), name="pipeline-watchdog")
    print("  [watchdog] pipeline watchdog scheduled (60s check, 90s threshold)", flush=True)

    # ── Server-ready preload: fire immediately on the event loop ─────────────
    # This fires the LLM preload as soon as _main() completes setup, giving
    # the model up to 10-30s of warm-up time before any client connects or
    # types their first message.  The pipeline thread fires its own preload
    # after warmup completes (may take 10-20s) — this fires first, in parallel.
    asyncio.create_task(
        _preload_model(LLM_CONFIG.model, keep_alive_s=120),
        name="server-ready-preload",
    )
    print("  [llm] background preload started at server ready", flush=True)

    threading.Thread(target=_stats_loop,           daemon=True).start()
    threading.Thread(target=_audio_ingestion_loop, daemon=True).start()
    threading.Thread(target=_pipeline_loop,        daemon=True).start()

    async with websockets.serve(_ws_handler, "localhost", 8765):
        print("  [ws] server listening on ws://localhost:8765")
        try:
            await asyncio.Future()   # blocks until cancelled (Ctrl-C)
        finally:
            # ── Graceful shutdown ─────────────────────────────────────────
            print("  [shutdown] cancelling active plan...", flush=True)
            await _cancel_plan()
            print("  [shutdown] cancelling pending reminders...", flush=True)
            await shutdown_all_reminders()
            print("  [shutdown] closing browser...", flush=True)
            try:
                if BrowserManager._instance:
                    await BrowserManager._instance.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nSmall O stopped.")
