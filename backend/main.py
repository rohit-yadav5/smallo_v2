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
from collections import Counter

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

# Active autonomous planner task.  Only one plan runs at a time;
# starting a new one cancels the previous.
_current_plan_task: asyncio.Task | None = None

# Continuous 60-second audio buffer — mic NEVER gated; set in _audio_ingestion_loop
_audio_buffer: RollingAudioBuffer | None = None

# Set by VAD oracle on barge-in; cleared by pipeline at start of each speaking phase
_interrupt_event: threading.Event = threading.Event()

# Current voice state — cached so new WS clients get it immediately
_current_voice_state: str = "idle"

# Partial bot response saved when user barge-ins mid-sentence.
# Injected into the next turn's LLM prompt so the bot knows where it left off.
_interrupted_partial: str = ""

# Protects _interrupted_partial across VAD callback and pipeline threads
_barge_in_lock: threading.Lock = threading.Lock()

# ── Planner management helpers ────────────────────────────────────────────────

# Words/phrases that cancel the running plan — checked BEFORE LLM call (fast).
_CANCEL_WORDS = frozenset([
    "stop", "cancel", "abort",
    "stop the task", "cancel the plan", "stop what you're doing",
    "stop that", "cancel that", "never mind",
])


async def _launch_plan(goal: str) -> None:
    """Start a new plan task on the main event loop, cancelling any active plan."""
    global _current_plan_task
    if _current_plan_task and not _current_plan_task.done():
        _current_plan_task.cancel()
        try:
            await _current_plan_task
        except asyncio.CancelledError:
            pass
    _current_plan_task = asyncio.create_task(
        run_plan(goal, _emit, max_steps=20),
        name=f"plan:{goal[:40]}",
    )
    print(f"  [plan] task created: {_current_plan_task.get_name()}", flush=True)


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
}

_MEMORY_IMPORTANCE = {
    "PersonalMemory":     8,
    "ConsolidatedMemory": 9,
    "DecisionMemory":     7,
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
        memories = retrieve_memories(user_text, top_k=10)
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
    for m in memories:
        summary = m.get("summary", "")
        mt      = m.get("memory_type", "")
        if mt == "ConsolidatedMemory":
            consolidated.append(summary)
        elif summary.lower().startswith(("user name", "user age", "user friend")):
            personal.append(summary)
        elif mt in ("ProjectMemory", "DecisionMemory", "ArchitectureMemory", "ActionMemory"):
            strategic.append(summary)
        elif mt == "ReflectionMemory":
            reflections.append(summary)

    ordered = consolidated[:2] + personal[:2] + strategic[:2] + reflections[:1]
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
        with tracker.step("TTS (plugin direct)"):
            try:
                speak(result["text"], _interrupt_event)
            except Exception as e:
                print(f"    [tts] plugin speak error: {e}")
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
                ai_text, tts_timing = speak_stream(
                    _token_broadcaster(ask_llm_stream(summary_prompt), _interrupt_event),
                    _interrupt_event,
                )
            except Exception as e:
                print(f"    [tts] plugin speak_stream error: {e}")
                return ""
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
        try:
            came_from_barge_in = _run_turn(turn, tracker, router, came_from_barge_in)
        except Exception as e:
            came_from_barge_in = False
            print(f"\n  [pipeline] !! UNHANDLED EXCEPTION in turn {turn}: {e}")
            import traceback; traceback.print_exc()
            _emit("VOICE_STATE", {"state": "idle"})
            time.sleep(1)   # brief pause before retrying


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

    with tracker.step("STT"):
        rec_start = time.perf_counter()
        _text_input_branch = False

        # Poll both text-input queue and speech-event queue.
        # 0.5s mini-timeout on speech keeps text input <500 ms latency.
        TIMEOUT_S = 30.0
        while True:
            # ── Text input has priority (zero-STT path) ────────────────
            try:
                user_text = _text_input_queue.get_nowait()
                trans_secs = 0.0
                rec_secs   = time.perf_counter() - rec_start
                _text_input_branch = True
                print(f"  [text_input] processing: {user_text[:80]!r}", flush=True)
                break
            except queue.Empty:
                pass

            # ── Speech event (normal voice path) ───────────────────────
            try:
                speech_start_s, speech_end_s = _speech_events.get(timeout=0.5)
                # STT branch — extract audio & transcribe
                audio_data = _audio_buffer.read_window(speech_start_s, speech_end_s)
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

            # ── Timeout guard ──────────────────────────────────────────
            if time.perf_counter() - rec_start >= TIMEOUT_S:
                print("  [pipeline] no input for 30 s — going idle")
                _emit("VOICE_STATE", {"state": "idle"})
                return False

    if not _text_input_branch:
        print(f"  [stt] wait={rec_secs:.3f}s  transcription={trans_secs:.3f}s")

    print(f"\n  {'You typed' if _text_input_branch else 'You said'}: {user_text}\n")

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

    _emit("VOICE_STATE", {"state": "thinking"})

    # ── Plan cancellation (fast path — no LLM call) ──────────────────────
    # Check before plugin routing so "stop" always works even during planning.
    _user_lower = user_text.lower().strip()
    if any(_user_lower == w or _user_lower.startswith(w + " ") for w in _CANCEL_WORDS):
        if _current_plan_task and not _current_plan_task.done():
            asyncio.run_coroutine_threadsafe(_cancel_plan(), _loop)
            _emit("VOICE_STATE", {"state": "speaking"})
            _interrupt_event.clear()
            try:
                speak("Stopped.", _interrupt_event)
            except Exception:
                pass
            _emit("VOICE_STATE", {"state": "idle"})
            tracker.summary()
            return False

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
    # Inject user context into every LLM call as a system-prompt suffix.
    user_ctx_suffix = get_context_prompt()

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
        # Launch plan on main asyncio loop (non-blocking from pipeline thread)
        asyncio.run_coroutine_threadsafe(_launch_plan(goal), _loop)
        # Immediately acknowledge verbally — no LLM call needed
        _emit("VOICE_STATE", {"state": "speaking"})
        _interrupt_event.clear()
        try:
            speak("On it. I'll let you know when it's done.", _interrupt_event)
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
            f"Piper synth first sentence: {tts_synth_s:.3f}s  ({n_sent} sentences)"
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

                    elif ev == "TEXT_INPUT":
                        # User typed a message — inject directly into pipeline.
                        text = ctrl.get("data", {}).get("text", "").strip()
                        if text:
                            _text_input_queue.put_nowait(text)
                            print(f"  [ws] TEXT_INPUT: {text[:80]!r}", flush=True)

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

    # ── Load persistent user context from disk ───────────────────────────────
    load_user_context()

    # Register the WS audio sender so TTS can stream audio to browser clients.
    # Must happen after _loop is set (sender uses run_coroutine_threadsafe).
    register_ws_audio_sender(_make_audio_sender())

    # ── Wire reminder tool's broadcast function ───────────────────────────────
    _set_reminder_broadcast(_emit)

    # ── Text-input watcher (file + WS both feed _text_input_queue) ─────────────
    def _on_text_transcript(text: str) -> None:
        """Enqueue typed text exactly like a finished STT result."""
        _text_input_queue.put_nowait(text)

    asyncio.create_task(watch_text_input(_on_text_transcript))

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


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nSmall O stopped.")
