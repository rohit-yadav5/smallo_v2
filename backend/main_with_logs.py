"""backend/main_with_logs.py – WebSocket server + voice pipeline (with detailed logging).

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
  VOICE_STATE    {state: idle | listening | thinking | speaking}
  STT_RESULT     {text, recording_time, transcription_time}
  LLM_TOKEN      {token, done}
  PLUGIN_ACTION  {plugin, action, result, direct}
  MEMORY_EVENT   {type, importance, summary, id, retrieved?}
  SYSTEM_STATS   {cpu, ram, battery}
"""
import asyncio
import json
import logging
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

# ──────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────

class _FuncFormatter(logging.Formatter):
    """Custom formatter: [TIMESTAMP] [LEVEL] [func_name] message"""
    def format(self, record: logging.LogRecord) -> str:
        ts    = self.formatTime(record, "%H:%M:%S.%f")[:-3]   # HH:MM:SS.mmm
        level = record.levelname[:5]                           # DEBUG/INFO /WARN /ERROR
        func  = record.funcName
        return f"[{ts}] [{level}] [{func}] {record.getMessage()}"

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_FuncFormatter())
_handler.setLevel(logging.DEBUG)

log = logging.getLogger("smallO")
log.setLevel(logging.DEBUG)
log.addHandler(_handler)
log.propagate = False

# ──────────────────────────────────────────────────
# High-quality audio resampling
# ──────────────────────────────────────────────────

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
    log.info("scipy resample_poly available — high-quality anti-aliased resampling active")
except ImportError:
    log.warning("scipy not available — falling back to linear interpolation resampling (lower quality)")
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
log.debug("BACKEND_DIR=%s", BACKEND_DIR)

from audio import RollingAudioBuffer
from stt import transcribe, transcribe_partial, warmup as stt_warmup, StreamingTranscriber
from vad import VADOracle
from llm import ask_llm_plugin_summary, warmup as llm_warmup
from tts import speak, speak_stream, warmup as tts_warmup, abort_speaking
from tts.main_tts import register_ws_audio_sender
from memory_system.retrieval.search import retrieve_memories
from memory_system.core.insert_pipeline import insert_memory
from plugins.router import PluginRouter
from utils.latency import LatencyTracker

log.info("All modules imported successfully")


# ──────────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_clients: set = set()

# Raw audio bytes from browser  [uint32 SR][float32[] samples]
_audio_queue: queue.Queue = queue.Queue()

# Speech event timestamps (start_s, end_s) from VADOracle → pipeline
_speech_events: queue.Queue = queue.Queue()

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

# Streaming STT — one StreamingTranscriber per utterance.
# It runs Whisper every 500 ms during speech (live word-by-word display)
# and produces the final result in finalize() at turn end.
def _make_transcriber() -> StreamingTranscriber:
    """Create a fresh StreamingTranscriber that emits STT_PARTIAL events."""
    log.debug("creating new StreamingTranscriber (chunk_interval=0.3s)")
    def _on_partial(confirmed: str, hypothesis: str) -> None:
        log.debug("STT_PARTIAL — confirmed=%r  hypothesis=%r", confirmed[:60], hypothesis[:40])
        _emit("STT_PARTIAL", {"text": confirmed, "hypothesis": hypothesis})
    t = StreamingTranscriber(
        transcribe_fn         = transcribe,
        on_partial            = _on_partial,
        chunk_interval_s      = 0.3,
        transcribe_partial_fn = transcribe_partial,
    )
    log.debug("StreamingTranscriber created: %s", t)
    return t

_active_transcriber: list = [None]   # [StreamingTranscriber | None]


def _emit(event: str, data: dict):
    """Thread-safe broadcast to all connected WebSocket clients."""
    global _current_voice_state
    if event == "VOICE_STATE":
        prev = _current_voice_state
        _current_voice_state = data.get("state", _current_voice_state)
        if prev != _current_voice_state:
            log.info("VOICE_STATE  %s → %s", prev, _current_voice_state)
    if not _loop or not _clients:
        log.debug("_emit %s — no loop or no clients, skipping ws broadcast", event)
        return

    async def _send_all():
        msg  = json.dumps({"event": event, "data": data})
        dead = set()
        for ws in list(_clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        if dead:
            log.warning("_emit %s — %d dead client(s) removed", event, len(dead))
        _clients.difference_update(dead)

    asyncio.run_coroutine_threadsafe(_send_all(), _loop)


def _make_audio_sender():
    """
    Build the WebSocket audio sender callback for main_tts.register_ws_audio_sender().

    Signature: fn(msg_dict | None, audio_bytes | None)
      fn(msg_dict, None)   → broadcast JSON event to all clients
      fn(None, raw_bytes)  → broadcast binary audio to all clients
    """
    def _sender(msg: dict | None, audio_bytes: bytes | None) -> None:
        if not _loop or not _clients:
            return

        if audio_bytes is not None:
            async def _send_binary():
                dead = set()
                for ws in list(_clients):
                    try:
                        await ws.send(audio_bytes)
                    except Exception:
                        dead.add(ws)
                if dead:
                    log.warning("_sender binary — %d dead client(s) removed", len(dead))
                _clients.difference_update(dead)
            asyncio.run_coroutine_threadsafe(_send_binary(), _loop)
        elif msg is not None:
            log.debug("_sender JSON  event=%s", msg.get("event"))
            async def _send_json():
                text = json.dumps(msg)
                dead = set()
                for ws in list(_clients):
                    try:
                        await ws.send(text)
                    except Exception:
                        dead.add(ws)
                if dead:
                    log.warning("_sender JSON %s — %d dead client(s) removed", msg.get("event"), len(dead))
                _clients.difference_update(dead)
            asyncio.run_coroutine_threadsafe(_send_json(), _loop)

    return _sender


# ──────────────────────────────────────────────────
# VAD oracle callbacks
# ──────────────────────────────────────────────────

def _on_speech_start(start_s: float) -> None:
    """Fired by VADOracle at confirmed onset (pre-buffer already applied)."""
    log.info("▶ speech start  t=%.3fs  (incl 2.0s pre-buffer)  state=%s",
             start_s, _current_voice_state)
    print(f"  [vad] ▶ speech start  t={start_s:.3f}s  (incl 2.0s pre-buffer)", flush=True)
    # Start a fresh StreamingTranscriber for live word display this turn.
    _active_transcriber[0] = _make_transcriber()
    log.debug("fresh StreamingTranscriber armed for speech at t=%.3fs", start_s)
    # Barge-in: if bot is speaking, abort immediately.
    if _current_voice_state == "speaking" and not _interrupt_event.is_set():
        log.info("⚡ BARGE-IN detected — calling abort_speaking()  speech_start=%.3fs", start_s)
        print(f"  [vad] ⚡ BARGE-IN  speech_start={start_s:.3f}s", flush=True)
        abort_speaking()
        _interrupt_event.set()
        log.debug("_interrupt_event set")
    else:
        log.debug("no barge-in (state=%s, interrupt_already=%s)",
                  _current_voice_state, _interrupt_event.is_set())


def _on_speech_end(start_s: float, end_s: float) -> None:
    """Fired by VADOracle at confirmed silence end (post-buffer already applied)."""
    dur = end_s - start_s
    log.info("■ speech end  [%.3fs → %.3fs]  dur=%.3fs  qsize_before=%d",
             start_s, end_s, dur, _speech_events.qsize())
    print(f"  [vad] ■ speech end  [{start_s:.3f}s → {end_s:.3f}s]  {dur:.3f}s", flush=True)
    _speech_events.put_nowait((start_s, end_s))
    log.debug("speech event enqueued  qsize_after=%d", _speech_events.qsize())


def _on_speech_chunk(chunk: np.ndarray) -> None:
    """Forward each 16 ms speech frame to StreamingTranscriber for live display."""
    t = _active_transcriber[0]
    if t is not None:
        t.feed(chunk)
    else:
        log.debug("_on_speech_chunk — no active transcriber, dropping chunk (%d samples)", len(chunk))


def _on_first_silence(snapshot: np.ndarray) -> None:
    """Start background Whisper at first silence (early-STT optimisation)."""
    t = _active_transcriber[0]
    log.info("first silence detected — snapshot=%d samples (%.2fs)  transcriber=%s",
             len(snapshot), len(snapshot) / 16_000, "present" if t else "None")
    if t is not None:
        t.start_finalize(snapshot)
        log.debug("start_finalize() called on transcriber")
    else:
        log.warning("_on_first_silence — no active transcriber, early-STT skipped")


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

    log.info("_audio_ingestion_loop starting")

    GRACE_S = 1.0   # seconds after "speaking" start to ignore VAD (TTS echo suppression)

    _audio_buffer = RollingAudioBuffer(capacity_s=60, sample_rate=16_000)
    log.info("RollingAudioBuffer created  capacity=60s  sr=16000")

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
    log.info("VADOracle created  onset=0.50  offset=0.35  onset_count=2  offset_count=45"
             "  pre_buffer=2.0s  post_buffer=2.0s")

    print("  [vad] VADOracle ready  (grace=1000ms  offset_count=45×16ms=720ms  onset_count=2  pre_buffer=2.0s  post_buffer=2.0s)", flush=True)

    _prev_state     : str   = ""
    _speaking_since : float = 0.0

    _frames_written  = 0   # DEBUG counter — logged every 500 frames
    _grace_skips     = 0   # DEBUG counter — frames skipped during grace period

    while True:
        try:
            raw = _audio_queue.get(timeout=1.0)
        except queue.Empty:
            log.debug("_audio_queue empty (1s timeout)  buffer_time=%.1fs",
                      _audio_buffer.current_time_s if _audio_buffer else 0.0)
            continue

        # ── Validate ────────────────────────────────────
        if len(raw) < 8:
            log.warning("received short packet len=%d — skipping", len(raw))
            continue
        src_sr = int(np.frombuffer(raw[:4], dtype=np.uint32)[0])
        if src_sr == 0 or src_sr > 192_000:
            log.warning("invalid sample rate %d — skipping packet", src_sr)
            continue

        # ── Decode & resample to 16 kHz ──────────────────
        samples = np.frombuffer(raw[4:], dtype=np.float32).copy()
        n_in    = len(samples)
        samples = _resample(samples, src_sr, 16_000)
        np.clip(samples, -1.0, 1.0, out=samples)

        _frames_written += 1
        if _frames_written % 500 == 0:
            log.debug("ingestion heartbeat  frames=%d  buffer_time=%.1fs  src_sr=%d  "
                      "samples_in=%d→%d  queue_depth=%d",
                      _frames_written, _audio_buffer.current_time_s,
                      src_sr, n_in, len(samples), _audio_queue.qsize())

        # ── ALWAYS write to rolling buffer — mic is NEVER gated ──────────
        _audio_buffer.write(samples)
        current_time = _audio_buffer.current_time_s

        state = _current_voice_state

        # ── Detect state transitions — reset LSTM to prevent bleed ───────
        if state != _prev_state:
            log.info("state transition  %s → %s  t=%.3fs", _prev_state, state, current_time)
            if state == "speaking":
                oracle.reset()
                _speaking_since = current_time
                _grace_skips    = 0
                log.info("oracle LSTM reset for speaking transition  grace_s=%.1f", GRACE_S)
                print("  [vad] → speaking  (LSTM reset, grace period armed)", flush=True)
            elif state == "listening":
                oracle.reset()
                log.info("oracle LSTM reset for listening transition  (from %s)", _prev_state)
                print(f"  [vad] → listening  (LSTM reset, from {_prev_state})", flush=True)
            _prev_state = state

        # ── Grace period during speaking: buffer receives audio but VAD ──
        # is silenced to prevent TTS echo from triggering false barge-in.
        if state == "speaking" and (current_time - _speaking_since) < GRACE_S:
            _grace_skips += 1
            if _grace_skips == 1 or _grace_skips % 50 == 0:
                log.debug("grace period active  elapsed=%.3fs / %.1fs  skips=%d",
                          current_time - _speaking_since, GRACE_S, _grace_skips)
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
    log.debug("extracting identity facts from: %r", user_text[:80])
    facts = []
    if m := re.search(r"my name is\s+([A-Za-z]+)", user_text, re.IGNORECASE):
        facts.append({"text": f"User name is {m.group(1)}", "memory_type": "PersonalMemory", "project_reference": "UserProfile"})
        log.info("identity: name detected → %s", m.group(1))
    if m := re.search(r"i am\s+(\d{1,3})", user_text, re.IGNORECASE):
        facts.append({"text": f"User age is {m.group(1)}", "memory_type": "PersonalMemory", "project_reference": "UserProfile"})
        log.info("identity: age detected → %s", m.group(1))
    if m := re.search(r"my friend'?s name is\s+([A-Za-z]+)", user_text, re.IGNORECASE):
        facts.append({"text": f"User friend's name is {m.group(1)}", "memory_type": "PersonalMemory", "project_reference": "UserProfile"})
        log.info("identity: friend name detected → %s", m.group(1))
    log.debug("identity extraction complete  facts=%d", len(facts))
    return facts


# ──────────────────────────────────────────────────
# Memory retrieval + context builder
# ──────────────────────────────────────────────────

def _build_memory_context(user_text: str) -> str:
    log.info("retrieving memories for: %r", user_text[:80])
    try:
        t0 = time.perf_counter()
        memories = retrieve_memories(user_text, top_k=10)
        log.info("memory retrieval done  count=%d  elapsed=%.3fs",
                 len(memories), time.perf_counter() - t0)
    except Exception as e:
        log.error("memory retrieval failed: %s", e, exc_info=True)
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
        log.debug("no memories retrieved")
        print("    [memory] none retrieved")
        return user_text

    type_counts = Counter(m.get("memory_type", "Unknown") for m in memories)
    log.info("memory types: %s", dict(type_counts))
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
    log.debug("memory context ordered  consolidated=%d  personal=%d  strategic=%d  reflections=%d  total_used=%d",
              len(consolidated), len(personal), len(strategic), len(reflections), len(ordered))
    if not ordered:
        log.debug("no ordered memories — returning bare user_text")
        return user_text

    context = "Relevant long-term memory:\n" + "\n".join(f"- {l}" for l in ordered)
    result  = f"{context}\n\nUser: {user_text}"
    log.info("memory context built  prompt_len=%d chars", len(result))
    return result


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
    log.debug("_token_broadcaster starting")
    _token_count = 0
    for token in token_gen:
        if interrupt_event and interrupt_event.is_set():
            log.info("_token_broadcaster interrupted at token %d — sending done=True", _token_count)
            _emit("LLM_TOKEN", {"token": "", "done": True})
            return
        _emit("LLM_TOKEN", {"token": token, "done": False})
        _token_count += 1
        if _token_count == 1:
            log.info("◌ first token received")
        yield token
    log.info("_token_broadcaster done  total_tokens=%d", _token_count)
    _emit("LLM_TOKEN", {"token": "", "done": True, "token_count": _token_count})


# ──────────────────────────────────────────────────
# Plugin helpers
# ──────────────────────────────────────────────────

def _handle_plugin_result(result: dict, tracker: LatencyTracker) -> str:
    log.info("plugin result  plugin=%s  action=%s  direct=%s  text_len=%d",
             result["plugin"], result["action"], result["direct"], len(result["text"]))
    _emit("PLUGIN_ACTION", {
        "plugin": result["plugin"],
        "action": result["action"],
        "result": result["text"][:200],
        "direct": result["direct"],
    })
    if result["direct"]:
        log.info("plugin direct TTS — speaking result directly")
        _emit("VOICE_STATE", {"state": "speaking"})
        _interrupt_event.clear()
        with tracker.step("TTS (plugin direct)"):
            try:
                speak(result["text"], _interrupt_event)
                log.info("plugin direct TTS complete")
            except Exception as e:
                log.error("plugin speak error: %s", e, exc_info=True)
                print(f"    [tts] plugin speak error: {e}")
        print(f"\n  Plugin: {result['text']}\n")
        return result["text"]
    else:
        log.info("plugin summarize path — sending to LLM + TTS")
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
                    _token_broadcaster(ask_llm_plugin_summary(summary_prompt), _interrupt_event),
                    _interrupt_event,
                )
                log.info("plugin summarize complete  first_word=%.3fs  total=%.3fs",
                         tts_timing["first_word_secs"], tts_timing["total_secs"])
            except Exception as e:
                log.error("plugin speak_stream error: %s", e, exc_info=True)
                print(f"    [tts] plugin speak_stream error: {e}")
                return ""
        print(f"    [tts] first word: {tts_timing['first_word_secs']:.3f}s  |  total: {tts_timing['total_secs']:.3f}s")
        print(f"\n  Plugin summary: {ai_text}\n")
        return ai_text


def _store_action_memory(user_text: str, spoken_text: str, result: dict):
    log.debug("scheduling action memory insert  plugin=%s  action=%s",
              result["plugin"], result["action"])
    def _run():
        log.debug("action memory insert starting  plugin=%s", result["plugin"])
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
                log.info("action memory inserted  id=%s  plugin=%s", memory_id, result["plugin"])
                _emit("MEMORY_EVENT", {
                    "retrieved": False,
                    "id":        memory_id,
                    "type":      "action",
                    "importance": _MEMORY_IMPORTANCE["ActionMemory"],
                    "summary":   f"{result['plugin']}: {spoken_text[:100]}",
                })
            else:
                log.warning("action memory insert returned no id  plugin=%s", result["plugin"])
        except Exception as e:
            log.error("action memory insert failed: %s", e, exc_info=True)
            print(f"    [memory] action memory insert failed: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ──────────────────────────────────────────────────
# System stats loop
# ──────────────────────────────────────────────────

def _stats_loop():
    log.info("_stats_loop starting")
    psutil.cpu_percent(interval=None)   # prime baseline (first call always returns 0.0)
    time.sleep(0.5)
    while True:
        try:
            battery = psutil.sensors_battery()
            cpu  = psutil.cpu_percent(interval=None)
            ram  = psutil.virtual_memory().percent
            bat  = round(battery.percent) if battery else 100
            log.debug("SYSTEM_STATS  cpu=%.1f%%  ram=%.1f%%  battery=%d%%", cpu, ram, bat)
            _emit("SYSTEM_STATS", {
                "cpu":     cpu,
                "ram":     ram,
                "battery": bat,
            })
        except Exception as e:
            log.warning("stats collection error: %s", e)
        time.sleep(2)


# ──────────────────────────────────────────────────
# Main pipeline loop
# ──────────────────────────────────────────────────

def _pipeline_loop():
    log.info("_pipeline_loop starting — warming up models")
    print("  Warming up models...", flush=True)
    t0 = time.perf_counter(); stt_warmup(); print(f"    STT  ready  ({time.perf_counter()-t0:.2f}s)", flush=True)
    log.info("STT warmup done  elapsed=%.2fs", time.perf_counter() - t0)
    t0 = time.perf_counter(); tts_warmup(); print(f"    TTS  ready  ({time.perf_counter()-t0:.2f}s)", flush=True)
    log.info("TTS warmup done  elapsed=%.2fs", time.perf_counter() - t0)
    t0 = time.perf_counter(); llm_warmup(); print(f"    LLM  ready  ({time.perf_counter()-t0:.2f}s)", flush=True)
    log.info("LLM warmup done  elapsed=%.2fs", time.perf_counter() - t0)

    print("\n  Loading plugins...")
    try:
        router = PluginRouter()
        log.info("PluginRouter loaded successfully")
    except Exception as e:
        log.error("PluginRouter failed to load: %s — continuing without plugins", e, exc_info=True)
        print(f"  [pipeline] plugin router failed to load: {e} — continuing without plugins")
        router = None
    print()

    turn               = 0
    came_from_barge_in = False
    log.info("entering main turn loop")
    while True:
        turn += 1
        tracker = LatencyTracker(turn=turn)
        log.info("─── Turn %d  (came_from_barge_in=%s) ───", turn, came_from_barge_in)
        print(f"\n{'─'*54}")
        print(f"  Turn {turn}")
        print(f"{'─'*54}")

        # ── Top-level crash guard — one bad turn never kills the loop ──────
        try:
            came_from_barge_in = _run_turn(turn, tracker, router, came_from_barge_in)
            log.info("Turn %d complete  came_from_barge_in(next)=%s", turn, came_from_barge_in)
        except Exception as e:
            came_from_barge_in = False
            log.error("UNHANDLED EXCEPTION in turn %d: %s", turn, e, exc_info=True)
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

    log.info("_run_turn start  turn=%d  came_from_barge_in=%s", turn, came_from_barge_in)

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
            log.info("cleared %d stale speech event(s) from previous turn", cleared)
            print(f"  [pipeline] cleared {cleared} stale speech event(s)")
        else:
            log.debug("no stale speech events to clear")
    else:
        log.info("came_from_barge_in=True — preserving pending speech events  qsize=%d",
                 _speech_events.qsize())

    # ── Wait for speech timestamps from VADOracle ─────────────────────────
    _emit("VOICE_STATE", {"state": "listening"})
    print(f"  [pipeline] VOICE_STATE → listening")
    log.info("VOICE_STATE → listening  (waiting for speech event)")

    with tracker.step("STT"):
        rec_start = time.perf_counter()
        print(f"  [stt] waiting for speech event...")
        log.debug("blocking on _speech_events.get(timeout=30)")
        try:
            speech_start_s, speech_end_s = _speech_events.get(timeout=30)
            log.info("speech event received  start=%.3fs  end=%.3fs  waited=%.3fs",
                     speech_start_s, speech_end_s, time.perf_counter() - rec_start)
        except queue.Empty:
            log.warning("no speech event in 30s — going idle")
            print("  [pipeline] no speech for 30 s — going idle (speak to wake)")
            _emit("VOICE_STATE", {"state": "idle"})
            return False

        # Extract audio from rolling buffer using VAD timestamps.
        # VADOracle already applied pre_buffer_s=2.0 and post_buffer_s=2.0,
        # so this window is guaranteed to include audio before VAD onset
        # (capturing the first word) and after VAD offset (last word).
        # No pre-roll stitching needed — the buffer is the single source of truth.
        log.debug("reading audio window from buffer  start=%.3fs  end=%.3fs", speech_start_s, speech_end_s)
        audio_data = _audio_buffer.read_window(speech_start_s, speech_end_s)

        # Guard: VAD can fire on a window that maps to no buffered frames yet.
        if audio_data is None or len(audio_data) == 0:
            log.warning("skipping transcription — empty audio window [%.3fs → %.3fs]",
                        speech_start_s, speech_end_s)
            print(
                f"  [stt] skipping transcription — empty audio window "
                f"[{speech_start_s:.3f}s → {speech_end_s:.3f}s]",
                flush=True,
            )
            _emit("VOICE_STATE", {"state": "idle"})
            return False

        dur      = len(audio_data) / 16_000
        rec_secs = time.perf_counter() - rec_start
        log.info("audio extracted  samples=%d  dur=%.2fs  waited=%.3fs  buffer_time=%.1fs",
                 len(audio_data), dur, rec_secs, _audio_buffer.current_time_s)

        print(
            f"  [stt] window [{speech_start_s:.3f}s → {speech_end_s:.3f}s]"
            f"  {dur:.2f}s audio  (waited {rec_secs:.3f}s)",
            flush=True,
        )
        tracker.note(f"utterance: {dur:.2f}s  [{speech_start_s:.3f}s → {speech_end_s:.3f}s]")

        # Transcribe — use streaming result if ready, else batch Whisper.
        transcriber             = _active_transcriber[0]
        _active_transcriber[0] = _make_transcriber()   # fresh for next turn
        log.debug("transcriber for this turn: %s  (fresh one armed for next)",
                  "StreamingTranscriber" if transcriber else "None→batch")

        try:
            if transcriber is not None:
                log.info("transcribing via StreamingTranscriber.finalize()  audio=%.2fs", dur)
                t_stt = time.perf_counter()
                user_text, trans_secs = transcriber.finalize(audio_data)
                log.info("streaming finalize done  text=%r  whisper=%.3fs  total_stt=%.3fs",
                         user_text[:80], trans_secs, time.perf_counter() - t_stt)
                print(f"  [stt] ✓ streaming finalize  ({trans_secs:.3f}s Whisper)", flush=True)
            else:
                log.info("transcribing via batch Whisper  audio=%.2fs", dur)
                t_stt = time.perf_counter()
                user_text, trans_secs = transcribe(audio_data)
                log.info("batch Whisper done  text=%r  whisper=%.3fs  total_stt=%.3fs",
                         user_text[:80], trans_secs, time.perf_counter() - t_stt)
        except Exception as e:
            log.error("transcription failed: %s", e, exc_info=True)
            print(f"  [stt] transcribe failed: {e}")
            _emit("VOICE_STATE", {"state": "idle"})
            return False
        tracker.note(f"transcript: '{user_text[:70]}{'…' if len(user_text)>70 else ''}'  ({trans_secs:.3f}s)")

    print(f"  [stt] wait={rec_secs:.3f}s  transcription={trans_secs:.3f}s")
    print(f"\n  You said: {user_text}\n")
    log.info("You said: %r  (wait=%.3fs  trans=%.3fs)", user_text, rec_secs, trans_secs)

    if not user_text.strip():
        log.warning("empty transcript — going idle")
        _emit("VOICE_STATE", {"state": "idle"})
        return False

    if user_text.lower().strip() in ("exit", "quit", "stop"):
        log.info("stop command received (%r) — going idle", user_text.lower().strip())
        _emit("VOICE_STATE", {"state": "idle"})
        return False

    _emit("STT_RESULT", {
        "text":               user_text,
        "recording_time":     round(rec_secs, 3),
        "transcription_time": round(trans_secs, 3),
    })

    _emit("VOICE_STATE", {"state": "thinking"})
    log.info("VOICE_STATE → thinking")

    # ── Plugin routing ───────────────────────────────────────────────────
    plugin_result = None
    if router is not None:
        log.debug("routing to plugin router  text=%r", user_text[:80])
        try:
            with tracker.step("Plugin Router"):
                plugin_result = router.route(user_text)
            if plugin_result:
                log.info("plugin match  plugin=%s  action=%s  direct=%s",
                         plugin_result["plugin"], plugin_result["action"], plugin_result["direct"])
            else:
                log.debug("no plugin matched")
        except Exception as e:
            log.error("plugin router error: %s", e, exc_info=True)
            print(f"  [plugin] router error: {e}")
    else:
        log.debug("router is None — skipping plugin routing")

    if plugin_result is not None:
        log.info("handling plugin result  plugin=%s", plugin_result["plugin"])
        try:
            spoken = _handle_plugin_result(plugin_result, tracker)
            _store_action_memory(user_text, spoken, plugin_result)
        except Exception as e:
            log.error("plugin handle result error: %s", e, exc_info=True)
            print(f"  [plugin] handle result error: {e}")
        _emit("VOICE_STATE", {"state": "idle"})
        tracker.summary()
        log.info("_run_turn done (plugin path)  turn=%d  → idle", turn)
        return False

    # ── Identity extraction ──────────────────────────────────────────────
    try:
        with tracker.step("Identity Extraction"):
            identity_facts = _extract_identity_facts(user_text)
        log.debug("identity extraction: %d facts", len(identity_facts))
    except Exception as e:
        log.error("identity extraction error: %s", e, exc_info=True)
        print(f"  [identity] extraction error: {e}")
        identity_facts = []

    try:
        with tracker.step("Identity Memory Insert"):
            for fact in identity_facts:
                log.debug("inserting identity fact: %s", fact["text"])
                memory_id = insert_memory(fact)
                if memory_id:
                    log.info("identity memory inserted  id=%s  text=%r", memory_id, fact["text"])
                    _emit("MEMORY_EVENT", {
                        "retrieved":  False,
                        "id":         memory_id,
                        "type":       "personal",
                        "importance": _MEMORY_IMPORTANCE["PersonalMemory"],
                        "summary":    fact["text"],
                    })
    except Exception as e:
        log.error("identity memory insert error: %s", e, exc_info=True)
        print(f"  [identity] memory insert error: {e}")

    # ── Memory retrieval ─────────────────────────────────────────────────
    try:
        with tracker.step("Memory Retrieval"):
            prompt = _build_memory_context(user_text)
        log.debug("prompt after memory context: %d chars", len(prompt))
    except Exception as e:
        log.error("memory context build error: %s", e, exc_info=True)
        print(f"  [memory] context build error: {e}")
        prompt = user_text

    # ── Interrupted-response context ─────────────────────────────────────
    # If the bot was cut off mid-response, prepend what it had already said
    # so it can acknowledge the interruption and seamlessly respond to the
    # new query (or pick up where it left off if the user asks it to).
    if _interrupted_partial:
        log.info("injecting interrupted context  chars=%d  snippet=%r",
                 len(_interrupted_partial), _interrupted_partial[:60])
        snip = _interrupted_partial[:400] + ("…" if len(_interrupted_partial) > 400 else "")
        prompt = (
            f"[Context: you were mid-response saying \"{snip}\" when the user interrupted you. "
            f"You don't need to repeat it; just respond naturally to what they said next.]\n\n"
            f"{prompt}"
        )
        print(f"  [pipeline] injecting interrupted context ({len(_interrupted_partial)} chars)")
        _interrupted_partial = ""
        log.debug("_interrupted_partial cleared  prompt now %d chars", len(prompt))

    # ── LLM + TTS ────────────────────────────────────────────────────────
    log.info("VOICE_STATE → speaking  prompt_chars=%d", len(prompt))
    print(f"  [pipeline] VOICE_STATE → speaking  |  prompt {len(prompt):,} chars", flush=True)
    _emit("VOICE_STATE", {"state": "speaking"})
    _interrupt_event.clear()   # arm: VAD can now set this on barge-in
    log.debug("_interrupt_event cleared — barge-in now armed")

    try:
        log.debug("starting speak_stream(ask_llm_plugin_summary(...))")
        t_llm_start = time.perf_counter()
        ai_text, tts_timing = speak_stream(
            _token_broadcaster(ask_llm_plugin_summary(prompt), _interrupt_event),
            _interrupt_event,
        )
        first_token = tts_timing.get("first_token_secs", tts_timing["first_word_secs"])
        first_audio = tts_timing["first_word_secs"]
        total_t     = tts_timing["total_secs"]
        n_tok       = tts_timing.get("token_count", "?")
        n_sent      = tts_timing.get("sentence_count", "?")
        tts_synth_s = max(0.0, first_audio - first_token)
        speaking_s  = max(0.0, total_t - first_audio)
        log.info("speak_stream done  first_token=%.3fs  first_audio=%.3fs  total=%.3fs"
                 "  tokens=%s  sentences=%s  tts_synth=%.3fs  speaking=%.3fs",
                 first_token, first_audio, total_t, n_tok, n_sent, tts_synth_s, speaking_s)
        log.info("AI said: %r", ai_text[:120])

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
        log.error("LLM/TTS error: %s", e, exc_info=True)
        print(f"  [llm/tts] error: {e}")
        _emit("VOICE_STATE", {"state": "idle"})
        tracker.summary()
        return

    # ── Barge-in: save partial response, decide next state ───────────────
    if _interrupt_event.is_set():
        log.info("interrupt_event is set — processing barge-in outcome  ai_text_len=%d",
                 len(ai_text.strip()))
        if ai_text.strip():
            # Bot had started speaking — real barge-in.  Save partial so the next
            # turn can acknowledge it, then listen for the user's follow-up.
            with _barge_in_lock:
                _interrupted_partial = ai_text.strip()
            chars = len(_interrupted_partial)
            snip  = _interrupted_partial[:80] + ("…" if chars > 80 else "")
            log.info("real barge-in — partial saved  chars=%d  snippet=%r", chars, snip)
            print(f"  [pipeline] ⚡ BARGE-IN — partial saved ({chars} chars): \"{snip}\"")
            print("  [pipeline] restarting turn from listening state")
            _emit("VOICE_STATE", {"state": "listening"})
            tracker.summary()
            log.info("_run_turn → True (barge-in, ai_text present)  turn=%d", turn)
            return True   # barge-in: next turn should NOT clear speech events
        else:
            # Bot hadn't spoken yet — could be a very quick interrupt or false
            # barge-in.  If a speech_end event is already queued, process it;
            # otherwise go idle.
            pending = not _speech_events.empty()
            log.info("barge-in with no ai_text  speech_event_pending=%s  qsize=%d",
                     pending, _speech_events.qsize())
            if pending:
                log.info("pending speech event — restarting turn")
                print("  [pipeline] ⚡ BARGE-IN — bot silent, speech event pending — restarting")
                _emit("VOICE_STATE", {"state": "listening"})
                tracker.summary()
                log.info("_run_turn → True (barge-in, no ai_text, pending event)  turn=%d", turn)
                return True
            log.warning("false barge-in (no text, no event) — going idle")
            print("  [pipeline] ⚡ False barge-in (no text, no event) — going idle")
            _emit("VOICE_STATE", {"state": "idle"})
            tracker.summary()
            log.info("_run_turn → False (false barge-in)  turn=%d", turn)
            return False

    _emit("VOICE_STATE", {"state": "idle"})
    tracker.summary()
    log.info("_run_turn → False (normal completion)  turn=%d", turn)

    # ── Reflection memory (background) ───────────────────────────────────
    def _store_reflection(ut=user_text, at=ai_text):
        log.debug("reflection memory insert starting  user=%r  ai=%r", ut[:40], at[:40])
        try:
            memory_id = insert_memory({
                "text":              f"User: {ut}\nAssistant: {at}",
                "memory_type":       "ReflectionMemory",
                "project_reference": "VoiceInteraction",
            })
            if memory_id:
                log.info("reflection memory inserted  id=%s", memory_id)
                _emit("MEMORY_EVENT", {
                    "retrieved":  False,
                    "id":         memory_id,
                    "type":       "reflection",
                    "importance": _MEMORY_IMPORTANCE["ReflectionMemory"],
                    "summary":    f"You: {ut[:50]}… / AI: {at[:50]}…",
                })
            else:
                log.warning("reflection memory insert returned no id")
        except Exception as e:
            log.error("reflection memory insert failed: %s", e, exc_info=True)
            print(f"  [memory] reflection insert failed: {e}")

    threading.Thread(target=_store_reflection, daemon=True).start()
    log.debug("reflection memory thread started")
    return False


# ──────────────────────────────────────────────────
# WebSocket server
# ──────────────────────────────────────────────────

async def _ws_handler(ws):
    _clients.add(ws)
    log.info("WS client connected  total=%d  addr=%s", len(_clients), ws.remote_address)
    print(f"  [ws] client connected  ({len(_clients)} total)  addr={ws.remote_address}")

    # Send current voice state immediately so late-connecting frontends sync up
    try:
        await ws.send(json.dumps({"event": "VOICE_STATE", "data": {"state": _current_voice_state}}))
        log.debug("sent initial VOICE_STATE=%s to new client", _current_voice_state)
    except Exception:
        pass

    try:
        async for message in ws:
            if isinstance(message, bytes):
                if len(message) < 8:
                    log.warning("short binary message len=%d — skipping", len(message))
                    continue
                src_sr = int(np.frombuffer(message[:4], dtype=np.uint32)[0])
                if src_sr == 0 or src_sr > 192_000:
                    log.warning("invalid sr=%d in binary message — skipping", src_sr)
                    continue
                _audio_queue.put_nowait(message)
                log.debug("audio queued  sr=%d  bytes=%d  queue_depth=%d",
                          src_sr, len(message), _audio_queue.qsize())
            else:
                try:
                    ctrl = json.loads(message)
                    if ctrl.get("event") == "ping":
                        log.debug("ping received — sending pong")
                        await ws.send(json.dumps({"event": "pong", "data": {}}))
                    else:
                        log.debug("control message: event=%s", ctrl.get("event"))
                except Exception:
                    pass
    except Exception as e:
        log.warning("WS handler exception (client likely disconnected): %s", e)
    finally:
        _clients.discard(ws)
        log.info("WS client disconnected  remaining=%d  addr=%s", len(_clients), ws.remote_address)
        print(f"  [ws] client disconnected  ({len(_clients)} remaining)")


async def _main():
    global _loop
    _loop = asyncio.get_running_loop()
    log.info("_main starting  loop=%s", _loop)

    register_ws_audio_sender(_make_audio_sender())
    log.info("WS audio sender registered")

    threading.Thread(target=_stats_loop,           daemon=True, name="stats").start()
    threading.Thread(target=_audio_ingestion_loop, daemon=True, name="audio_ingestion").start()
    threading.Thread(target=_pipeline_loop,        daemon=True, name="pipeline").start()
    log.info("threads started: stats, audio_ingestion, pipeline")

    async with websockets.serve(_ws_handler, "localhost", 8765):
        log.info("WebSocket server listening on ws://localhost:8765")
        print("  [ws] server listening on ws://localhost:8765")
        await asyncio.Future()


if __name__ == "__main__":
    log.info("Small O starting up")
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down")
        print("\nSmall O stopped.")
