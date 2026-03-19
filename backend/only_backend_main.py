import sys
import time
import threading
from pathlib import Path
import re
from collections import Counter

from utils.latency import LatencyTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stt import listen, warmup as stt_warmup
from llm import ask_llm_stream, warmup as llm_warmup
from tts import speak, speak_stream, warmup as tts_warmup

from memory_system.retrieval.search import retrieve_memories
from memory_system.core.insert_pipeline import insert_memory

from plugins.router import PluginRouter


# -----------------------------
# Identity Extraction
# -----------------------------
def extract_identity_facts(user_text: str):
    facts = []

    name_match = re.search(r"my name is\s+([A-Za-z]+)", user_text, re.IGNORECASE)
    if name_match:
        facts.append({
            "text": f"User name is {name_match.group(1)}",
            "memory_type": "PersonalMemory",
            "project_reference": "UserProfile"
        })

    age_match = re.search(r"i am\s+(\d{1,3})", user_text, re.IGNORECASE)
    if age_match:
        facts.append({
            "text": f"User age is {age_match.group(1)}",
            "memory_type": "PersonalMemory",
            "project_reference": "UserProfile"
        })

    friend_match = re.search(r"my friend'?s name is\s+([A-Za-z]+)", user_text, re.IGNORECASE)
    if friend_match:
        facts.append({
            "text": f"User friend's name is {friend_match.group(1)}",
            "memory_type": "PersonalMemory",
            "project_reference": "UserProfile"
        })

    return facts


# -----------------------------
# Memory Context Builder
# -----------------------------
def build_memory_context(user_text: str) -> str:
    memories = retrieve_memories(user_text, top_k=10)

    if not memories:
        print("    [memory] none retrieved")
        return user_text

    # Log what was retrieved
    type_counts = Counter(m.get("memory_type", "Unknown") for m in memories)
    detail = "  ".join(f"{t}:{n}" for t, n in type_counts.most_common())
    print(f"    [memory] {len(memories)} retrieved  —  {detail}")

    consolidated = []
    personal = []
    strategic = []
    reflections = []

    for m in memories:
        summary = m.get("summary", "")
        memory_type = m.get("memory_type", "")

        if memory_type == "ConsolidatedMemory":
            consolidated.append(summary)
        elif (summary.lower().startswith("user name")
              or summary.lower().startswith("user age")
              or summary.lower().startswith("user friend's name")):
            personal.append(summary)
        elif memory_type in ["ProjectMemory", "DecisionMemory", "ArchitectureMemory",
                              "ActionMemory"]:
            strategic.append(summary)
        elif memory_type == "ReflectionMemory":
            reflections.append(summary)

    ordered = (
        consolidated[:2] +
        personal[:2] +
        strategic[:2] +
        reflections[:1]
    )

    if not ordered:
        return user_text

    context = "Relevant long-term memory:\n" + "\n".join(f"- {line}" for line in ordered)
    return f"{context}\n\nUser: {user_text}"


# -----------------------------
# Plugin Helpers
# -----------------------------
def _handle_plugin_result(result: dict, tracker: LatencyTracker) -> str:
    """Speak a plugin result — directly or via LLM summarization."""
    if result["direct"]:
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
        with tracker.step("LLM + TTS (plugin summarize)"):
            ai_text, tts_timing = speak_stream(ask_llm_stream(summary_prompt))
        print(f"    [tts] first word: {tts_timing['first_word_secs']:.3f}s  |  total: {tts_timing['total_secs']:.3f}s")
        print(f"\n  Plugin summary: {ai_text}\n")
        return ai_text


def _store_action_memory(user_text: str, spoken_text: str, result: dict):
    """Persist a plugin action as ActionMemory in a background thread."""
    threading.Thread(
        target=insert_memory,
        args=({
            "text": (
                f"User requested: {user_text}\n"
                f"Plugin: {result['plugin']} / Action: {result['action']}\n"
                f"Response: {spoken_text}"
            ),
            "memory_type": "ActionMemory",
            "project_reference": f"Plugin:{result['plugin']}",
            "source": "plugin"
        },),
        daemon=True
    ).start()


# -----------------------------
# Main Loop
# -----------------------------
def run():
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

        with tracker.step("STT"):
            user_text, rec_secs, trans_secs = listen()
        print(f"    [stt] speaking: {rec_secs:.3f}s  |  transcription: {trans_secs:.3f}s")

        print(f"\n  You said: {user_text}\n")

        if not user_text:
            continue

        if user_text.lower().strip() in ["exit", "quit", "stop"]:
            print("Exiting Small O.")
            break

        # ─────────────────────────────────────────────────────────────
        # Plugin routing — fast keyword dispatch before memory/LLM
        # ─────────────────────────────────────────────────────────────
        with tracker.step("Plugin Router"):
            plugin_result = router.route(user_text)

        if plugin_result is not None:
            spoken = _handle_plugin_result(plugin_result, tracker)
            _store_action_memory(user_text, spoken, plugin_result)
            tracker.summary()
            continue  # Skip memory retrieval and LLM for this turn
        # ─────────────────────────────────────────────────────────────

        # Normal flow: identity extraction → memory → LLM → TTS
        with tracker.step("Identity Extraction"):
            identity_facts = extract_identity_facts(user_text)
            if identity_facts:
                print(f"    [identity] {len(identity_facts)} fact(s) found")

        with tracker.step("Identity Memory Insert"):
            for fact in identity_facts:
                insert_memory(fact)

        with tracker.step("Memory Retrieval"):
            prompt = build_memory_context(user_text)

        with tracker.step("LLM + TTS"):
            ai_text, tts_timing = speak_stream(ask_llm_stream(prompt))
        print(f"    [tts] first word: {tts_timing['first_word_secs']:.3f}s  |  total: {tts_timing['total_secs']:.3f}s")

        print(f"\n  AI: {ai_text}\n")

        tracker.summary()

        # Store reflection memory in background (non-blocking)
        threading.Thread(
            target=insert_memory,
            args=({
                "text": f"User: {user_text}\nAssistant: {ai_text}",
                "memory_type": "ReflectionMemory",
                "project_reference": "VoiceInteraction"
            },),
            daemon=True
        ).start()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nSmall O stopped by user.")
