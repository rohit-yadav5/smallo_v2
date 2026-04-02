import json
import requests
import re

from llm.SYSTEM_PROMPT import SYSTEM_PROMPT

# =====================
# OLLAMA CONFIG
# =====================

OLLAMA_URL = "http://localhost:11434/api/generate"
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

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{safe_text}\n\n"
        "Assistant:"
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "num_predict": -1
            }
        },
        timeout=60
    )

    response.raise_for_status()
    return response.json()["response"].strip()


# =====================
# STREAMING LLM CALL
# =====================

_LLM_OPTIONS = {
    "num_predict": -1,
    "stop": ["User:", "Human:"]
}


def warmup():
    """Check Ollama health, then warm the model into RAM."""
    # Health check — surfaces problems early with clear instructions
    if not check_ollama():
        print("  [llm] ⚠  Ollama not ready — first real turn may be slow or fail", flush=True)
        return
    # Send a 1-token request so phi3 is fully paged into RAM
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": "Hi.", "stream": False,
                  "keep_alive": -1, "options": {"num_predict": 1}},
            timeout=30
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
        print(f"  [llm] ✗ Cannot connect to Ollama at {OLLAMA_URL} — "
              f"is it running?  (run: ollama serve)", flush=True)
        return False
    except Exception as e:
        print(f"  [llm] ✗ Ollama health check error: {e}", flush=True)
        return False


def ask_llm_stream(user_text: str):
    """Yield response tokens one at a time as they arrive from Ollama."""
    safe_text = _sanitize_user_text(user_text)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{safe_text}\n\n"
        "Assistant:"
    )

    print(f"  [llm] ▶ {len(prompt):,} char prompt → {MODEL}", flush=True)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "keep_alive": -1,
            "options": _LLM_OPTIONS
        },
        stream=True,
        # (connect_timeout, read_timeout_per_chunk)
        # read_timeout applies to each iter_lines() call — prevents infinite hang
        # if Ollama stops mid-generation.
        timeout=(10, 90),
    )

    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done", False):
                yield data.get("response", "")
