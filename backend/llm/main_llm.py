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
    """Send a 1-token request so Ollama pages phi3 into RAM before first real turn."""
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": "Hi.", "stream": False,
                  "options": {"num_predict": 1}},
            timeout=30
        )
    except Exception:
        pass  # Ollama not running yet — will load on first real request


def ask_llm_stream(user_text: str):
    """Yield response tokens one at a time as they arrive from Ollama."""
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
            "stream": True,
            "options": _LLM_OPTIONS
        },
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done", False):
                yield data.get("response", "")
