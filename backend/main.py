"""backend/main.py – WebSocket server + voice pipeline with Silero VAD.

Audio flow:
  Browser AudioWorklet → WebSocket binary chunks → _audio_queue
  → _vad_loop (Silero VAD, resample) → _speech_queue (complete utterances)
  → _pipeline_loop (Whisper → LLM → TTS)

Barge-in flow:
  _vad_loop detects speech while TTS playing
  → abort_speaking() (cuts audio immediately)
  → _interrupt_event.set()
  → _pipeline_loop restarts turn from listening state

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
import queue
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import websockets

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from stt import transcribe, warmup as stt_warmup
from stt.vad import StreamingVAD
from llm import ask_llm_stream, warmup as llm_warmup
from tts import speak, speak_stream, warmup as tts_warmup, abort_speaking
from memory_system.retrieval.search import retrieve_memories
from memory_system.core.insert_pipeline import insert_memory
from plugins.router import PluginRouter
from utils.latency import LatencyTracker


# ──────────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_clients: set = set()

# Raw audio bytes from browser  [uint32 SR][float32[] samples]
_audio_queue: queue.Queue = queue.Queue()

# Complete VAD-processed utterances (float32 @ 16kHz) → pipeline
_speech_queue: queue.Queue = queue.Queue()

# Set by VAD loop on barge-in; cleared by pipeline at start of each speaking phase
_interrupt_event: threading.Event = threading.Event()

# Current voice state — cached so new WS clients get it immediately
_current_voice_state: str = "idle"


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


# ──────────────────────────────────────────────────
# VAD loop — runs continuously in its own thread
# ──────────────────────────────────────────────────

def _vad_loop():
    """
    Read raw audio chunks from _audio_queue, feed to Silero VAD.

    Behaviour by pipeline state:
      listening → accumulate speech; on complete utterance → _speech_queue
      speaking  → detect speech onset → abort TTS + set _interrupt_event
      idle/thinking → process to keep pre-speech buffer warm; discard output
    """
    vad = StreamingVAD(
        onset_threshold  = 0.50,
        offset_threshold = 0.35,
        silence_ms       = 600,
        min_speech_ms    = 250,
        pre_pad_ms       = 200,
    )

    print("  [vad] Silero VAD ready")

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

        # ── Decode & resample to 16 kHz ─────────────────
        samples = np.frombuffer(raw[4:], dtype=np.float32).copy()
        if src_sr != 16_000:
            n       = int(len(samples) * 16_000 / src_sr)
            samples = np.interp(
                np.linspace(0, len(samples), n),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)

        state = _current_voice_state

        # ── Barge-in detection (TTS speaking) ────────────
        if state == "speaking":
            utterance = vad.process(samples)
            if vad.is_speaking and not _interrupt_event.is_set():
                print("  [vad] ⚡ BARGE-IN — user spoke during TTS")
                abort_speaking()          # cut audio immediately
                _interrupt_event.set()   # signal pipeline to restart
                vad.reset()              # clean state for next listen session
            continue

        # ── Normal speech capture (pipeline listening) ───
        if state == "listening":
            utterance = vad.process(samples)
            if utterance is not None and len(utterance) > 0:
                dur = len(utterance) / 16_000
                print(f"  [vad] → utterance queued  {dur:.2f}s")
                _speech_queue.put_nowait(utterance)
            continue

        # ── Idle / thinking — keep pre-speech buffer warm ─
        vad.process(samples)


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

def _token_broadcaster(token_gen):
    """Wrap an LLM generator: emit LLM_TOKEN events while yielding to TTS."""
    for token in token_gen:
        _emit("LLM_TOKEN", {"token": token, "done": False})
        yield token
    _emit("LLM_TOKEN", {"token": "", "done": True})


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
                    _token_broadcaster(ask_llm_stream(summary_prompt)),
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
    print("  Warming up models...")
    t0 = time.perf_counter(); stt_warmup(); print(f"    STT  ready  ({time.perf_counter()-t0:.2f}s)")
    t0 = time.perf_counter(); tts_warmup(); print(f"    TTS  ready  ({time.perf_counter()-t0:.2f}s)")
    t0 = time.perf_counter(); llm_warmup(); print(f"    LLM  ready  ({time.perf_counter()-t0:.2f}s)")

    print("\n  Loading plugins...")
    try:
        router = PluginRouter()
    except Exception as e:
        print(f"  [pipeline] plugin router failed to load: {e} — continuing without plugins")
        router = None
    print()

    turn = 0
    while True:
        turn += 1
        tracker = LatencyTracker(turn=turn)
        print(f"\n{'─'*54}")
        print(f"  Turn {turn}")
        print(f"{'─'*54}")

        # ── Top-level crash guard — one bad turn never kills the loop ──────
        try:
            _run_turn(turn, tracker, router)
        except Exception as e:
            print(f"\n  [pipeline] !! UNHANDLED EXCEPTION in turn {turn}: {e}")
            import traceback; traceback.print_exc()
            _emit("VOICE_STATE", {"state": "idle"})
            time.sleep(1)   # brief pause before retrying


def _run_turn(turn: int, tracker: LatencyTracker, router):
    """Execute one full pipeline turn. Called inside the crash-guard try/except."""

    # ── Clear stale complete utterances from previous turn ───────────────
    cleared = 0
    while not _speech_queue.empty():
        try:
            _speech_queue.get_nowait()
            cleared += 1
        except queue.Empty:
            break
    if cleared:
        print(f"  [pipeline] cleared {cleared} stale utterance(s) from speech queue")

    # ── Wait for a complete utterance from the VAD loop ───────────────────
    _emit("VOICE_STATE", {"state": "listening"})
    print(f"  [pipeline] VOICE_STATE → listening")

    with tracker.step("STT"):
        rec_start = time.perf_counter()
        print(f"  [stt] waiting for utterance from VAD...")
        try:
            audio_data = _speech_queue.get(timeout=120)   # float32 @ 16kHz
        except queue.Empty:
            print("  [pipeline] no speech for 120s — re-entering listen loop")
            _emit("VOICE_STATE", {"state": "idle"})
            return
        rec_secs = time.perf_counter() - rec_start
        dur      = len(audio_data) / 16_000

        tracker.note(f"utterance: {dur:.2f}s  {len(audio_data)} samples @ 16kHz")
        print(f"  [stt] utterance received after {rec_secs:.3f}s — {dur:.2f}s")

        try:
            user_text, trans_secs = transcribe(audio_data)
        except Exception as e:
            print(f"  [stt] transcribe failed: {e}")
            _emit("VOICE_STATE", {"state": "idle"})
            return
        tracker.note(f"transcript: '{user_text[:70]}{'…' if len(user_text)>70 else ''}'  ({trans_secs:.3f}s)")

    print(f"  [stt] wait={rec_secs:.3f}s  transcription={trans_secs:.3f}s")
    print(f"\n  You said: {user_text}\n")

    if not user_text.strip():
        _emit("VOICE_STATE", {"state": "idle"})
        return

    if user_text.lower().strip() in ("exit", "quit", "stop"):
        _emit("VOICE_STATE", {"state": "idle"})
        return

    _emit("STT_RESULT", {
        "text":               user_text,
        "recording_time":     round(rec_secs, 3),
        "transcription_time": round(trans_secs, 3),
    })

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
        return

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

    # ── LLM + TTS ────────────────────────────────────────────────────────
    _emit("VOICE_STATE", {"state": "speaking"})
    _interrupt_event.clear()   # arm: VAD can now set this on barge-in

    try:
        with tracker.step("LLM + TTS"):
            ai_text, tts_timing = speak_stream(
                _token_broadcaster(ask_llm_stream(prompt)),
                _interrupt_event,
            )
        print(f"    [tts] first word: {tts_timing['first_word_secs']:.3f}s  |  total: {tts_timing['total_secs']:.3f}s")
        print(f"\n  AI: {ai_text}\n")
    except Exception as e:
        print(f"  [llm/tts] error: {e}")
        _emit("VOICE_STATE", {"state": "idle"})
        tracker.summary()
        return

    # ── Barge-in: restart from listening immediately ──────────────────────
    if _interrupt_event.is_set():
        print("  [pipeline] barge-in detected — restarting from listening")
        _emit("VOICE_STATE", {"state": "listening"})
        tracker.summary()
        return   # next loop iteration emits listening again at top

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
                    if ctrl.get("event") == "ping":
                        await ws.send(json.dumps({"event": "pong", "data": {}}))
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

    threading.Thread(target=_stats_loop,    daemon=True).start()
    threading.Thread(target=_vad_loop,      daemon=True).start()
    threading.Thread(target=_pipeline_loop, daemon=True).start()

    async with websockets.serve(_ws_handler, "localhost", 8765):
        print("  [ws] server listening on ws://localhost:8765")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nSmall O stopped.")
