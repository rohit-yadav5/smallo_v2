"""backend/main.py – WebSocket server bridging the voice pipeline to the frontend.

Events emitted (JSON: {"event": "...", "data": {...}}):
  VOICE_STATE    {state: idle | listening | thinking | speaking}
  STT_RESULT     {text, recording_time, transcription_time}
  LLM_TOKEN      {token, done}
  PLUGIN_ACTION  {plugin, action, result}
  MEMORY_EVENT   {type, importance, summary, id, retrieved?}
  SYSTEM_STATS   {cpu, ram, battery}
"""
import asyncio
import json
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import psutil
import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stt import listen, warmup as stt_warmup
from llm import ask_llm_stream, warmup as llm_warmup
from tts import speak, speak_stream, warmup as tts_warmup
from memory_system.retrieval.search import retrieve_memories
from memory_system.core.insert_pipeline import insert_memory
from plugins.router import PluginRouter
from utils.latency import LatencyTracker


# ──────────────────────────────────────────────────
# WebSocket state
# ──────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_clients: set = set()


def _emit(event: str, data: dict):
    """Thread-safe broadcast to all connected WebSocket clients."""
    if not _loop or not _clients:
        return

    async def _send_all():
        msg = json.dumps({"event": event, "data": data})
        dead = set()
        for ws in list(_clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        _clients.difference_update(dead)

    asyncio.run_coroutine_threadsafe(_send_all(), _loop)


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
    memories = retrieve_memories(user_text, top_k=10)

    for m in memories:
        _emit("MEMORY_EVENT", {
            "retrieved": True,
            "id": m["memory_id"],
            "type": _MEMORY_TYPE_MAP.get(m["memory_type"], "idea"),
            "importance": round(m["score"] * 10, 1),
            "summary": m.get("summary", ""),
        })

    if not memories:
        print("    [memory] none retrieved")
        return user_text

    type_counts = Counter(m.get("memory_type", "Unknown") for m in memories)
    print(f"    [memory] {len(memories)} retrieved  —  " + "  ".join(f"{t}:{n}" for t, n in type_counts.most_common()))

    consolidated, personal, strategic, reflections = [], [], [], []
    for m in memories:
        summary = m.get("summary", "")
        mt = m.get("memory_type", "")
        if mt == "ConsolidatedMemory":
            consolidated.append(summary)
        elif summary.lower().startswith(("user name", "user age", "user friend's name")):
            personal.append(summary)
        elif mt in ["ProjectMemory", "DecisionMemory", "ArchitectureMemory", "ActionMemory"]:
            strategic.append(summary)
        elif mt == "ReflectionMemory":
            reflections.append(summary)

    ordered = consolidated[:2] + personal[:2] + strategic[:2] + reflections[:1]
    if not ordered:
        return user_text

    context = "Relevant long-term memory:\n" + "\n".join(f"- {line}" for line in ordered)
    return f"{context}\n\nUser: {user_text}"


# ──────────────────────────────────────────────────
# Token broadcaster — intercepts LLM tokens for WS
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
    })
    if result["direct"]:
        _emit("VOICE_STATE", {"state": "speaking"})
        with tracker.step("TTS (plugin direct)"):
            speak(result["text"])
        print(f"\n  Plugin: {result['text']}\n")
        return result["text"]
    else:
        summary_prompt = (
            "You are Small O. Summarize the following data in 1-2 friendly spoken sentences. "
            "Be concise and natural. Highlight the most useful information.\n\n"
            f"Data:\n{result['text']}"
        )
        _emit("VOICE_STATE", {"state": "speaking"})
        with tracker.step("LLM + TTS (plugin summarize)"):
            ai_text, tts_timing = speak_stream(_token_broadcaster(ask_llm_stream(summary_prompt)))
        print(f"    [tts] first word: {tts_timing['first_word_secs']:.3f}s  |  total: {tts_timing['total_secs']:.3f}s")
        print(f"\n  Plugin summary: {ai_text}\n")
        return ai_text


def _store_action_memory(user_text: str, spoken_text: str, result: dict):
    def _run():
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
                "id": memory_id,
                "type": "action",
                "importance": _MEMORY_IMPORTANCE["ActionMemory"],
                "summary": f"{result['plugin']}: {spoken_text[:100]}",
            })
    threading.Thread(target=_run, daemon=True).start()


# ──────────────────────────────────────────────────
# System stats loop
# ──────────────────────────────────────────────────

def _stats_loop():
    while True:
        try:
            battery = psutil.sensors_battery()
            _emit("SYSTEM_STATS", {
                "cpu": psutil.cpu_percent(interval=None),
                "ram": psutil.virtual_memory().percent,
                "battery": round(battery.percent) if battery else 100,
            })
        except Exception:
            pass
        time.sleep(2)


# ──────────────────────────────────────────────────
# Main pipeline loop (blocking — runs in a thread)
# ──────────────────────────────────────────────────

def _pipeline_loop():
    print("  Warming up models...")
    t0 = time.perf_counter()
    stt_warmup()
    print(f"    STT  ready  ({time.perf_counter() - t0:.2f}s)")
    t0 = time.perf_counter()
    tts_warmup()
    print(f"    TTS  ready  ({time.perf_counter() - t0:.2f}s)")
    t0 = time.perf_counter()
    llm_warmup()
    print(f"    LLM  ready  ({time.perf_counter() - t0:.2f}s)")

    print("\n  Loading plugins...")
    router = PluginRouter()
    print()

    turn = 0
    while True:
        turn += 1
        tracker = LatencyTracker()
        print(f"\n{'─' * 54}")
        print(f"  Turn {turn}")
        print(f"{'─' * 54}")

        # ── Listen ──────────────────────────────────────
        _emit("VOICE_STATE", {"state": "listening"})
        with tracker.step("STT"):
            user_text, rec_secs, trans_secs = listen()
        print(f"    [stt] speaking: {rec_secs:.3f}s  |  transcription: {trans_secs:.3f}s")
        print(f"\n  You said: {user_text}\n")

        if not user_text:
            _emit("VOICE_STATE", {"state": "idle"})
            continue

        if user_text.lower().strip() in ["exit", "quit", "stop"]:
            _emit("VOICE_STATE", {"state": "idle"})
            break

        _emit("STT_RESULT", {
            "text": user_text,
            "recording_time": round(rec_secs, 3),
            "transcription_time": round(trans_secs, 3),
        })

        _emit("VOICE_STATE", {"state": "thinking"})

        # ── Plugin routing ───────────────────────────────
        with tracker.step("Plugin Router"):
            plugin_result = router.route(user_text)

        if plugin_result is not None:
            spoken = _handle_plugin_result(plugin_result, tracker)
            _store_action_memory(user_text, spoken, plugin_result)
            _emit("VOICE_STATE", {"state": "idle"})
            tracker.summary()
            continue

        # ── Identity extraction ──────────────────────────
        with tracker.step("Identity Extraction"):
            identity_facts = _extract_identity_facts(user_text)
            if identity_facts:
                print(f"    [identity] {len(identity_facts)} fact(s) found")

        with tracker.step("Identity Memory Insert"):
            for fact in identity_facts:
                memory_id = insert_memory(fact)
                if memory_id:
                    _emit("MEMORY_EVENT", {
                        "retrieved": False,
                        "id": memory_id,
                        "type": "personal",
                        "importance": _MEMORY_IMPORTANCE["PersonalMemory"],
                        "summary": fact["text"],
                    })

        # ── Memory retrieval ─────────────────────────────
        with tracker.step("Memory Retrieval"):
            prompt = _build_memory_context(user_text)

        # ── LLM + TTS ────────────────────────────────────
        _emit("VOICE_STATE", {"state": "speaking"})
        with tracker.step("LLM + TTS"):
            ai_text, tts_timing = speak_stream(_token_broadcaster(ask_llm_stream(prompt)))
        print(f"    [tts] first word: {tts_timing['first_word_secs']:.3f}s  |  total: {tts_timing['total_secs']:.3f}s")
        print(f"\n  AI: {ai_text}\n")

        _emit("VOICE_STATE", {"state": "idle"})
        tracker.summary()

        # ── Reflection memory (background) ───────────────
        def _store_reflection(ut=user_text, at=ai_text):
            memory_id = insert_memory({
                "text": f"User: {ut}\nAssistant: {at}",
                "memory_type": "ReflectionMemory",
                "project_reference": "VoiceInteraction",
            })
            if memory_id:
                _emit("MEMORY_EVENT", {
                    "retrieved": False,
                    "id": memory_id,
                    "type": "reflection",
                    "importance": _MEMORY_IMPORTANCE["ReflectionMemory"],
                    "summary": f"You: {ut[:50]}… / AI: {at[:50]}…",
                })

        threading.Thread(target=_store_reflection, daemon=True).start()


# ──────────────────────────────────────────────────
# WebSocket server
# ──────────────────────────────────────────────────

async def _ws_handler(ws):
    _clients.add(ws)
    print(f"  [ws] client connected  ({len(_clients)} total)")
    try:
        await ws.wait_closed()
    finally:
        _clients.discard(ws)
        print(f"  [ws] client disconnected  ({len(_clients)} remaining)")


async def _main():
    global _loop
    _loop = asyncio.get_running_loop()

    threading.Thread(target=_stats_loop, daemon=True).start()
    threading.Thread(target=_pipeline_loop, daemon=True).start()

    async with websockets.serve(_ws_handler, "localhost", 8765):
        print("  [ws] server listening on ws://localhost:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nSmall O stopped.")
