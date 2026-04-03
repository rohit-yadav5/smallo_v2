import json
import requests
import re

from llm.SYSTEM_PROMPT import SYSTEM_PROMPT

# =====================
# OLLAMA CONFIG
# =====================

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL     = "http://localhost:11434/api/chat"
MODEL = "phi3"

# =====================
# INPUT SANITIZATION
# =====================

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
    """Remove obvious prompt-injection attempts while keeping user intent."""
    clean_lines = []
    for line in text.splitlines():
        lower = line.strip().lower()
        if any(re.search(p, lower) for p in _INJECTION_PATTERNS):
            continue
        clean_lines.append(line)
    return " ".join(clean_lines).strip()


# =====================
# LLM CALL
# =====================

def ask_llm(user_text: str) -> str:
    safe_text = _sanitize_user_text(user_text)

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model":      MODEL,
            "messages":   [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": safe_text},
            ],
            "stream":     False,
            "keep_alive": -1,
            "options":    {"num_predict": 150},
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# =====================
# STREAMING LLM CALL
# =====================

_LLM_OPTIONS = {
    "num_predict": 150,          # 1–2 spoken sentences ≈ 40–80 tokens; 150 gives headroom
    "stop":        ["User:", "Human:"],
}


def warmup():
    """Check Ollama health, then warm the model into RAM."""
    if not check_ollama():
        print("  [llm] ⚠  Ollama not ready — first real turn may be slow or fail", flush=True)
        return
    # Use the chat endpoint so the system-prompt KV cache is seeded on the very
    # first real turn (generate would seed a different cache slot).
    try:
        requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model":      MODEL,
                "messages":   [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": "Hi."},
                ],
                "stream":     False,
                "keep_alive": -1,
                "options":    {"num_predict": 1},
            },
            timeout=30,
        )
        print(f"  [llm] ✓ model '{MODEL}' warmed up", flush=True)
    except Exception as e:
        print(f"  [llm] ⚠  warmup request failed: {e}", flush=True)


def check_ollama() -> bool:
    """
    Quick health probe: confirm Ollama is reachable and the model is loaded.
    Returns True if ready, False otherwise.  Never raises.
    """
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code != 200:
            print(f"  [llm] ✗ Ollama health check failed (HTTP {r.status_code})", flush=True)
            return False
        tags = r.json().get("models", [])
        names = [m.get("name", "") for m in tags]
        if not any(MODEL in n for n in names):
            print(f"  [llm] ✗ Model '{MODEL}' not found in Ollama. "
                  f"Available: {names}  →  run: ollama pull {MODEL}", flush=True)
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(f"  [llm] ✗ Cannot connect to Ollama at {OLLAMA_CHAT_URL} — "
              f"is it running?  (run: ollama serve)", flush=True)
        return False
    except Exception as e:
        print(f"  [llm] ✗ Ollama health check error: {e}", flush=True)
        return False


def ask_llm_stream(user_text: str):
    """Yield response tokens one at a time as they arrive from Ollama."""
    safe_text = _sanitize_user_text(user_text)

    # Split memory context (prefix) from the raw user utterance so that Ollama
    # can place them in appropriate message slots.  _build_memory_context formats
    # the combined string as:  "<context>\n\nUser: <utterance>"
    if "\n\nUser: " in safe_text:
        memory_ctx, utterance = safe_text.split("\n\nUser: ", 1)
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{memory_ctx}"},
            {"role": "user",   "content": utterance},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": safe_text},
        ]

    total_chars = sum(len(m["content"]) for m in messages)
    print(f"  [llm] ▶ {total_chars:,} char prompt → {MODEL}", flush=True)

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model":      MODEL,
            "messages":   messages,
            "stream":     True,
            "keep_alive": -1,
            "options":    _LLM_OPTIONS,
        },
        stream=True,
        # (connect_timeout, read_timeout_per_chunk)
        timeout=(10, 90),
    )

    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done", False):
                yield data.get("message", {}).get("content", "")
