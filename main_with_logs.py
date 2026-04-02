"""main_with_logs.py – Small O with full diagnostic logging.

Streams timestamped, colour-coded stdout+stderr from both the backend
pipeline and the Vite frontend.  Use this when debugging audio/connection
issues instead of the silent main.py.

Pipeline:
  Browser mic → AudioWorklet → WebSocket → Silero VAD → Whisper → LLM → Piper TTS

Barge-in:
  User speech during TTS cuts playback immediately (Pa_AbortStream).
  The pre-speech ring buffer (200ms) is preserved so the interrupting words
  are included.  The partial bot response is injected as context into the
  next LLM call so the bot knows where it left off.

Usage:
    python main_with_logs.py
"""
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
BACKEND  = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV_PY  = ROOT / ".venv" / "bin" / "python3"
PYTHON   = str(VENV_PY) if VENV_PY.exists() else sys.executable

# ── ANSI codes ────────────────────────────────────────────────────────
R    = "\033[0m"
B    = "\033[1m"
DIM  = "\033[2m"
UL   = "\033[4m"
GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
MAG  = "\033[95m"
BLU  = "\033[94m"
WHT  = "\033[97m"
ORG  = "\033[33m"   # orange (plugin)
PRP  = "\033[35m"   # purple (memory)
TL   = "\033[38;5;51m"   # teal (VAD)

# ── Per-subsystem tag → ANSI colour ──────────────────────────────────
_TAG_COLORS = {
    "vad":      TL,
    "stt":      BLU,
    "tts":      YLW,
    "llm":      CYN,
    "pipeline": WHT,
    "ws":       DIM,
    "memory":   PRP,
    "plugin":   ORG,
    "identity": PRP,
    "latency":  DIM,
}

_SESSION_START = time.perf_counter()


# ── Helpers ───────────────────────────────────────────────────────────
def _ts() -> str:
    """Current wall-clock time as HH:MM:SS.mmm"""
    return datetime.now().strftime("%H:%M:%S.%f")[:12]

def _session_elapsed() -> str:
    """Seconds since main_with_logs.py was launched."""
    e = time.perf_counter() - _SESSION_START
    m, s = divmod(int(e), 60)
    return f"{m:02d}:{s:02d}" if m else f"{s:5.1f}s"

def _hi(text: str, col: str, pattern: str = r"(\d+\.?\d*)s\b") -> str:
    """Highlight numbers matching `pattern` in white within a coloured line."""
    return re.sub(pattern, f"{WHT}\\1s{col}", text)


def _banner():
    print()
    print(f"  {WHT}{B}╔══════════════════════════════════════════════════╗{R}")
    print(f"  {WHT}{B}║          Small O  ·  Diagnostic Mode            ║{R}")
    print(f"  {WHT}{B}╚══════════════════════════════════════════════════╝{R}")
    print()
    print(f"  {DIM}Python   : {PYTHON}{R}")
    print(f"  {DIM}Backend  : {BACKEND}{R}")
    print(f"  {DIM}Frontend : {FRONTEND}{R}")
    print()
    print(f"  {DIM}Pipeline config:{R}")
    print(f"    {DIM}STT  : faster-whisper  base.en  int8  (beam=5  vad_filter=True){R}")
    print(f"    {DIM}VAD  : Silero  grace=1000ms  onset_count=3  step=256  min_speech=120ms{R}")
    print(f"    {DIM}TTS  : Piper  en_US-amy-medium  length_scale=0.7{R}")
    print(f"    {DIM}LLM  : Ollama  phi3  (stream  timeout=10/60s){R}")
    print()
    print(f"  {DIM}Log colour legend:{R}")
    print(f"    {TL}■{R} VAD      {BLU}■{R} STT      {YLW}■{R} TTS      {CYN}■{R} LLM")
    print(f"    {PRP}■{R} Memory   {ORG}■{R} Plugin   {MAG}■{R} Barge-in {RED}■{R} Error")
    print()


def _section(title: str, color: str = WHT):
    w   = 50
    pad = (w - len(title) - 2) // 2
    print(f"\n  {color}{B}{'─' * pad} {title} {'─' * pad}{R}\n")


# ── Main line formatter ───────────────────────────────────────────────

def _format_backend_line(line: str) -> str:
    """
    Return a fully formatted string for one backend log line.

    Handles:
      ┌── Turn separators     (──── Turn N ────)
      ├── User transcript      (You said: …)
      ├── Bot response         (  AI: …)
      ├── Barge-in / false     (⚡ BARGE-IN … / ⚡ False barge-in …)
      ├── [vad]  tagged        (▶ speech start prob=X / ■ speech end / ✗ too short /
      │                         → utterance queued|captured / → speaking|listening)
      ├── [stt]  tagged        (◌ waiting / ● received / timing)
      ├── [tts]  tagged        (⚙ synth #N Xs→Xs / ■ first_word/total/tokens)
      ├── [llm]  tagged        (▶ prompt / ✓ first token / ✓ warmed up /
      │                         ⚠ not ready / ◌ waiting / ✗ timeout / ■ throughput)
      ├── [pipeline] tagged    (◆ VOICE_STATE → / prompt chars / false barge-in /
      │                         injecting interrupted / no speech / cleared stale)
      ├── [memory] tagged      (🧠 retrieved / 💾 insert)
      ├── [plugin] tagged      (⚙ routed / action)
      ├── [ws]   tagged        (connect / disconnect — dim)
      ├── Latency tracker      (▶ Step start / ✓ Step Xs)
      ├── Warmup / ready lines
      └── Errors / tracebacks
    """
    stripped = line.strip()
    ll       = line.lower()
    ts       = _ts()
    indent   = "  "

    # ── Hard errors / exceptions ─────────────────────────────────────
    if any(k in ll for k in ("error", "exception", "traceback", "!! unhandled",
                              "failed", "crash", "✗", "hung?")):
        # Keep "too short" VAD discards and early-STT fallbacks out of the error bucket
        if "too short" not in ll and "falling back" not in ll:
            return f"{indent}{DIM}{ts}{R}  {RED}{B}{stripped}{R}"

    # ── Warnings ─────────────────────────────────────────────────────
    if any(k in ll for k in ("warn", "warning")):
        return f"{indent}{DIM}{ts}{R}  {YLW}{stripped}{R}"

    # ── Turn separator  ───  Turn N  ─────────────────────────────────
    if "─" in stripped and re.search(r"turn\s+\d+", ll):
        m        = re.search(r"turn\s+(\d+)", ll)
        turn_num = m.group(1) if m else "?"
        bar      = "━" * 54
        return (
            f"\n{indent}{WHT}{B}{bar}{R}\n"
            f"{indent}{WHT}{B}  TURN {turn_num}{R}"
            f"  {DIM}@ {ts}  +{_session_elapsed()}{R}\n"
            f"{indent}{WHT}{B}{bar}{R}\n"
        )

    # ── "You said:" — user transcript ────────────────────────────────
    if "you said:" in ll:
        text = re.sub(r"(?i)you said:\s*", "", stripped)
        return (
            f"\n{indent}{DIM}{ts}{R}  "
            f"{CYN}{B}┌─ YOU ──────────────────────────────────{R}\n"
            f"{indent}             {CYN}{B}│ {WHT}{B}{text}{R}\n"
            f"{indent}             {CYN}{B}└────────────────────────────────────────{R}\n"
        )

    # ── "  AI: …" — bot response ──────────────────────────────────────
    if re.match(r"\s+ai:\s+", line, re.IGNORECASE):
        text = re.sub(r"(?i)^\s*ai:\s*", "", stripped)
        return (
            f"\n{indent}{DIM}{ts}{R}  "
            f"{GRN}{B}┌─ BOT ──────────────────────────────────{R}\n"
            f"{indent}             {GRN}{B}│ {WHT}{B}{text}{R}\n"
            f"{indent}             {GRN}{B}└────────────────────────────────────────{R}\n"
        )

    # ── Barge-in / interrupt events ───────────────────────────────────
    if any(k in line for k in ("⚡", "BARGE-IN", "barge-in", "barge_in",
                                "partial saved", "injecting interrupted",
                                "barge-in utterance", "restarting turn",
                                "False barge-in", "bot silent")):
        # False barge-in (no real speech) shown dimmer — it's noise, not a real event
        is_false = "false barge-in" in ll or "false barge" in ll
        col = f"{MAG}{DIM}" if is_false else f"{MAG}{B}"
        return f"{indent}{DIM}{ts}{R}  {col}⚡  {stripped}{R}"

    # ── Latency tracker lines (▶ Step / ✓ Step) ──────────────────────
    if re.match(r"\s*[▶✓]\s", stripped):
        is_start  = stripped.startswith("▶")
        step_col  = DIM if not is_start else WHT
        sym       = f"{WHT}▶{R}" if is_start else f"{GRN}✓{R}"
        formatted = re.sub(r"(\d+\.?\d*)(s\b)", f"{WHT}\\1\\2{DIM}", stripped)
        return f"{indent}{DIM}{ts}{R}  {step_col}{formatted}{R}"

    # ── [subsystem] tagged lines ──────────────────────────────────────
    m = re.search(r"\[(\w+)\]", line)
    if m:
        tag = m.group(1).lower()
        col = _TAG_COLORS.get(tag, WHT)

        # ── [vad] ────────────────────────────────────────────────────
        if tag == "vad":
            # ▶ speech start  prob=0.xxx
            if "speech start" in ll:
                prob = re.search(r"prob=([\d.]+)", line)
                prob_val = float(prob.group(1)) if prob else 0.0
                # Colour prob by confidence: green ≥0.8, yellow ≥0.6, default otherwise
                prob_col = (GRN if prob_val >= 0.8 else YLW if prob_val >= 0.6 else TL)
                prob_str = f"  prob={prob_col}{prob.group(1)}{TL}" if prob else ""
                return f"{indent}{DIM}{ts}{R}  {TL}{B}▶ {stripped}{prob_str}{R}"
            # ■ speech end / forced
            if "speech end" in ll or "speech forced" in ll or (
                    "■" in line and "speech" in ll):
                formatted = _hi(stripped, TL)
                return f"{indent}{DIM}{ts}{R}  {TL}■ {formatted}{R}"
            # ✗ too short — discarded (dim, not an error)
            if "too short" in ll or "discarding" in ll:
                return f"{indent}{DIM}{ts}{R}  {DIM}✗ {stripped}{R}"
            # → utterance queued / barge-in utterance captured
            if "utterance queued" in ll or "utterance captured" in ll:
                formatted = _hi(stripped, TL)
                return f"{indent}{DIM}{ts}{R}  {TL}{B}→ {formatted}{R}"
            # ● Silero VAD ready (startup)
            if "silero vad ready" in ll:
                return f"{indent}{DIM}{ts}{R}  {TL}● {stripped}{R}"
            # → speaking / → listening  (LSTM reset state transitions)
            if "→ speaking" in ll or "→ listening" in ll:
                state_col = GRN if "speaking" in ll else BLU
                return f"{indent}{DIM}{ts}{R}  {TL}◈  {state_col}{stripped}{R}"
            return f"{indent}{DIM}{ts}{R}  {col}{stripped}{R}"

        # ── [stt] ────────────────────────────────────────────────────
        if tag == "stt":
            if "waiting" in ll:
                return f"{indent}{DIM}{ts}{R}  {BLU}◌ {stripped}{R}"
            if "utterance received" in ll or "using barge-in" in ll:
                formatted = _hi(stripped, BLU)
                return f"{indent}{DIM}{ts}{R}  {BLU}{B}● {formatted}{R}"
            # Early STT start — fired at first silence frame
            if "early stt started" in ll or "early transcription started" in ll:
                formatted = _hi(stripped, BLU)
                return f"{indent}{DIM}{ts}{R}  {BLU}{B}▶ {formatted}{R}"
            # Early STT result used (saves full Whisper time from critical path)
            if "early result ready" in ll:
                formatted = _hi(stripped, BLU)
                return f"{indent}{DIM}{ts}{R}  {BLU}{B}✓ {formatted}{R}"
            # Early STT fallback — warning, not error
            if "early result failed" in ll or "falling back" in ll:
                return f"{indent}{DIM}{ts}{R}  {YLW}⚠  {stripped}{R}"
            formatted = _hi(stripped, BLU)
            return f"{indent}{DIM}{ts}{R}  {BLU}{formatted}{R}"

        # ── [tts] ────────────────────────────────────────────────────
        if tag == "tts":
            # [tts] synth #N  Xs → Xs audio  'sentence'
            if "synth" in ll:
                # Highlight the timing numbers
                formatted = re.sub(r"(\d+\.?\d+)s", f"{WHT}\\1s{YLW}", stripped)
                return f"{indent}{DIM}{ts}{R}  {YLW}⚙  {formatted}{R}"
            # [tts] first_word=Xs  total=Xs  tokens=N  sentences=N
            if "first_word" in ll or "first word" in ll:
                formatted = re.sub(r"(\d+\.?\d+)s", f"{WHT}\\1s{YLW}", stripped)
                formatted = re.sub(r"(tokens|sentences)=(\d+)", f"\\1={WHT}\\2{YLW}", formatted)
                return f"{indent}{DIM}{ts}{R}  {YLW}■ {formatted}{R}"
            # [tts] plugin speak or other
            formatted = re.sub(r"(\d+\.?\d+)s", f"{WHT}\\1s{YLW}", stripped)
            return f"{indent}{DIM}{ts}{R}  {YLW}{formatted}{R}"

        # ── [llm] ────────────────────────────────────────────────────
        if tag == "llm":
            # [llm] ▶ N char prompt → model
            if ("prompt" in ll and "char" in ll) or (
                    "▶" in line and "prompt" in ll):
                chars_m = re.search(r"([\d,]+)\s+char", stripped)
                chars   = chars_m.group(1) if chars_m else "?"
                model_m = re.search(r"→\s+(\S+)$", stripped)
                model   = model_m.group(1) if model_m else ""
                return (f"{indent}{DIM}{ts}{R}  "
                        f"{CYN}▶  prompt {WHT}{chars}{CYN} chars"
                        f"{f'  →  {WHT}{model}{CYN}' if model else ''}{R}")

            # [llm] ✓ first token  Xs  (runtime — bold cyan, timing highlighted)
            if "first token" in ll:
                formatted = re.sub(r"(\d+\.?\d+)s", f"{WHT}\\1s{CYN}", stripped)
                return f"{indent}{DIM}{ts}{R}  {CYN}{B}✓  {formatted}{R}"

            # [llm] ✓ model 'phi3' warmed up  (startup — green, not timing-critical)
            if "warmed up" in ll or ("✓" in line and "model" in ll):
                model_m = re.search(r"'([^']+)'", stripped)
                model   = model_m.group(1) if model_m else ""
                return (f"{indent}{DIM}{ts}{R}  "
                        f"{GRN}✓  LLM model {WHT}{model}{GRN} warmed up{R}")

            # [llm] ⚠  Ollama not ready / warmup failed
            if "⚠" in line or "not ready" in ll or "warmup" in ll:
                return f"{indent}{DIM}{ts}{R}  {YLW}{B}⚠  {stripped}{R}"

            # [llm] ◌ waiting for first token...  Ns
            if "◌" in line or "waiting for first token" in ll:
                elapsed_m = re.search(r"(\d+)s$", stripped)
                elapsed   = elapsed_m.group(1) if elapsed_m else "?"
                return (f"{indent}{DIM}{ts}{R}  "
                        f"{DIM}{CYN}◌  waiting for first token...  {WHT}{elapsed}s{DIM}{R}")

            # [llm] ✗ no tokens / timeout / error  (already caught at top by error
            # handler, but kept here as a safety net for any that slip through)
            if "✗" in line or "no tokens" in ll or "hung" in ll:
                return f"{indent}{DIM}{ts}{R}  {RED}{B}✗  {stripped}{R}"

            # [llm] ■ N tokens  X tok/s  first_word=Xs  total=Xs
            if "■" in line or "tok/s" in ll:
                fmt = re.sub(r"(\d+\.?\d*)\s+tok/s", f"{WHT}\\1{CYN} tok/s", stripped)
                fmt = re.sub(r"(\d+)\s+tokens",       f"{WHT}\\1{CYN} tokens",  fmt)
                fmt = re.sub(r"(\d+\.?\d+)s",         f"{WHT}\\1s{CYN}",        fmt)
                return f"{indent}{DIM}{ts}{R}  {CYN}■  {fmt}{R}"

            # producer started / other dim [llm] lines
            return f"{indent}{DIM}{ts}{R}  {DIM}{CYN}{stripped}{R}"

        # ── [pipeline] ───────────────────────────────────────────────
        if tag == "pipeline":
            if "voice_state" in ll:
                # Colour the state value
                state_m = re.search(r"→\s+(\w+)", stripped)
                state   = state_m.group(1) if state_m else ""
                _state_col = {
                    "listening": BLU, "thinking": YLW,
                    "speaking":  GRN, "idle":     DIM,
                }.get(state, WHT)
                extra = stripped.split("|")[1].strip() if "|" in stripped else ""
                return (f"{indent}{DIM}{ts}{R}  {WHT}◆  VOICE_STATE → "
                        f"{_state_col}{B}{state}{R}"
                        f"{f'  {DIM}{extra}' if extra else ''}{R}")
            if "prompt" in ll and "chars" in ll:
                formatted = re.sub(r"([\d,]+)\s+chars", f"{WHT}\\1{WHT} chars", stripped)
                return f"{indent}{DIM}{ts}{R}  {WHT}◆  {formatted}{R}"
            if "cleared" in ll and "stale" in ll:
                return f"{indent}{DIM}{ts}{R}  {DIM}{stripped}{R}"
            if "injecting interrupted" in ll:
                return f"{indent}{DIM}{ts}{R}  {MAG}{stripped}{R}"
            if "no speech for" in ll:
                return f"{indent}{DIM}{ts}{R}  {YLW}⏱  {stripped}{R}"
            # False barge-in — dim magenta (it's noise, not a real interrupt)
            if "false barge-in" in ll:
                return f"{indent}{DIM}{ts}{R}  {MAG}{DIM}⚡  {stripped}{R}"
            # Real barge-in with or without partial — bright magenta
            if "barge-in" in ll or "barge_in" in ll:
                return f"{indent}{DIM}{ts}{R}  {MAG}{B}⚡  {stripped}{R}"
            return f"{indent}{DIM}{ts}{R}  {WHT}{stripped}{R}"

        # ── [memory] ─────────────────────────────────────────────────
        if tag == "memory":
            if "retrieved" in ll and "none" in ll:
                return f"{indent}{DIM}{ts}{R}  {DIM}🧠 {stripped}{R}"
            if "retrieved" in ll:
                return f"{indent}{DIM}{ts}{R}  {PRP}{B}🧠 {stripped}{R}"
            if "insert" in ll or "stored" in ll:
                return f"{indent}{DIM}{ts}{R}  {PRP}💾 {stripped}{R}"
            return f"{indent}{DIM}{ts}{R}  {PRP}{DIM}{stripped}{R}"

        # ── [ws] — dim (very noisy) ───────────────────────────────────
        if tag == "ws":
            if "connected" in ll:
                return f"{indent}{DIM}{ts}  ⚡ {stripped}{R}"
            if "disconnected" in ll:
                return f"{indent}{DIM}{ts}  ✕ {stripped}{R}"
            return f"{indent}{DIM}{ts}  {stripped}{R}"

        # ── [plugin] ─────────────────────────────────────────────────
        if tag == "plugin":
            return f"{indent}{DIM}{ts}{R}  {ORG}{B}⚙  {stripped}{R}"

        # ── [identity] ───────────────────────────────────────────────
        if tag == "identity":
            return f"{indent}{DIM}{ts}{R}  {PRP}👤 {stripped}{R}"

        # Generic tagged line
        return f"{indent}{DIM}{ts}{R}  {col}{stripped}{R}"

    # ── Latency: timing / secs mentions without [tag] ────────────────
    if any(k in ll for k in (" secs", " ms ", "latency", "tok/s")):
        formatted = re.sub(r"(\d+\.?\d*)(s\b)", f"{WHT}\\1\\2{DIM}", stripped)
        return f"{indent}{DIM}{ts}  {formatted}{R}"

    # ── Warmup / ready / started ──────────────────────────────────────
    if any(k in ll for k in ("ready", "✓ ", "connected", "started",
                              "listening on", "warming up", "loading")):
        return f"{indent}{DIM}{ts}{R}  {GRN}{stripped}{R}"

    # ── Plugin summary / action result lines ─────────────────────────
    if re.match(r"\s+(plugin|plugin summary):", ll):
        return f"{indent}{DIM}{ts}{R}  {ORG}{stripped}{R}"

    # ── Default: dim timestamp, plain line ───────────────────────────
    return f"{indent}{DIM}{ts}{R}  {stripped}"


def _stream_reader(pipe, label: str, col: str, stop_event: threading.Event,
                   is_backend: bool = False):
    """Read lines from a subprocess pipe and print with colour-coded prefix."""
    prefix = f"{col}{B}[{label:<8}]{R}"
    try:
        for raw in pipe:
            if stop_event.is_set():
                break
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            if is_backend:
                print(_format_backend_line(line), flush=True)
            else:
                # Frontend (Vite) — simpler treatment
                ll = line.lower()
                ts = _ts()
                if any(k in ll for k in ("error", "failed")):
                    lc = RED
                elif "warn" in ll:
                    lc = YLW
                elif any(k in ll for k in ("ready", "localhost", "✓", "hmr")):
                    lc = GRN
                else:
                    lc = DIM
                print(f"  {DIM}{ts}{R}  {prefix} {lc}{line.strip()}{R}", flush=True)
    except Exception:
        pass


# ── Health checks ──────────────────────────────────────────────────────
def _check_port_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def _preflight():
    ok = True
    print(f"  {B}Pre-flight checks{R}")

    # Python venv
    py_ok = Path(PYTHON).exists()
    sym   = f"{GRN}✓{R}" if py_ok else f"{RED}✗{R}"
    print(f"    {sym}  Python venv  {DIM}{PYTHON}{R}")
    if not py_ok:
        print(f"       {RED}→ run:  python -m venv .venv && pip install -r requirements.txt{R}")
        ok = False

    # Node modules
    nm    = FRONTEND / "node_modules"
    nm_ok = nm.exists()
    sym   = f"{GRN}✓{R}" if nm_ok else f"{RED}✗{R}"
    print(f"    {sym}  node_modules  {DIM}{nm}{R}")
    if not nm_ok:
        print(f"       {RED}→ run:  cd frontend && npm install{R}")
        ok = False

    # Piper voice model (TTS)
    voice_path = Path.home() / "piper-voices" / "en_US-amy-medium.onnx"
    voice_ok   = voice_path.exists()
    sym        = f"{GRN}✓{R}" if voice_ok else f"{YLW}!{R}"
    note       = "" if voice_ok else f"  {YLW}← TTS will fail{R}"
    print(f"    {sym}  Piper voice  {DIM}{voice_path}{R}{note}")

    # faster-whisper base.en model cache
    whisper_cache = (Path.home() / ".cache" / "huggingface" / "hub"
                     / "models--Systran--faster-whisper-base.en")
    whisper_ok = whisper_cache.exists()
    sym  = f"{GRN}✓{R}" if whisper_ok else f"{YLW}!{R}"
    note = ("" if whisper_ok
            else f"  {YLW}← will auto-download on first run (~150 MB){R}")
    print(f"    {sym}  Whisper base.en  {DIM}{whisper_cache}{R}{note}")

    # scipy (high-quality resampler)
    try:
        import scipy  # noqa: F401
        scipy_ok = True
    except ImportError:
        scipy_ok = False
    sym  = f"{GRN}✓{R}" if scipy_ok else f"{YLW}!{R}"
    note = ("" if scipy_ok
            else f"  {YLW}← fallback linear resampler active (lower audio quality){R}")
    print(f"    {sym}  scipy  {DIM}(polyphase resampler){R}{note}")

    # Ports free
    for port, name in [(8765, "WebSocket :8765"), (5173, "Vite      :5173")]:
        free = _check_port_free(port)
        sym  = f"{GRN}✓{R}" if free else f"{YLW}!{R}"
        note = "" if free else f"  {YLW}← busy, may conflict{R}"
        print(f"    {sym}  {name}{note}")

    print()
    return ok


# ── Kill helpers ────────────────────────────────────────────────────────
def _kill_tree(proc: subprocess.Popen, label: str) -> None:
    try:
        import psutil
        parent   = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:   child.terminate()
            except psutil.NoSuchProcess: pass
        try:   parent.terminate()
        except psutil.NoSuchProcess: pass
        _, alive = psutil.wait_procs([parent] + children, timeout=3)
        for p in alive:
            try:   p.kill()
            except psutil.NoSuchProcess: pass
        print(f"  {GRN}✓{R}  {label} stopped")
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=5)
            print(f"  {GRN}✓{R}  {label} stopped")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"  {YLW}!{R}  {label} killed (didn't stop in time)")
        except Exception:
            pass


# ── Main ────────────────────────────────────────────────────────────────
def main():
    global _SESSION_START
    _SESSION_START = time.perf_counter()

    _banner()
    if not _preflight():
        print(f"  {RED}Pre-flight failed — fix the issues above and retry.{R}\n")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"]       = str(BACKEND)
    env["FORCE_COLOR"]      = "1"
    env["PYTHONUTF8"]       = "1"
    # Critical: disable Python's block-buffering on stdout/stderr when running
    # as a subprocess.  Without this, backend print() calls accumulate in an
    # 8 KB buffer and never appear in the log until the buffer flushes — making
    # the system look frozen even when it is working normally.
    env["PYTHONUNBUFFERED"] = "1"

    procs: list[subprocess.Popen] = []
    stop  = threading.Event()
    backend_proc  = None
    frontend_proc = None

    try:
        # ── Backend ───────────────────────────────────────────────────
        _section("BACKEND", MAG)
        backend_proc = subprocess.Popen(
            [PYTHON, "-u", "main.py"],   # -u: unbuffered (belt+suspenders with PYTHONUNBUFFERED)
            cwd=BACKEND,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append(backend_proc)
        threading.Thread(
            target=_stream_reader,
            args=(backend_proc.stdout, "BACKEND", MAG, stop, True),
            daemon=True,
        ).start()
        print(f"  {MAG}{B}[BACKEND]{R}  pid={backend_proc.pid}  python={PYTHON}")

        # ── Frontend ──────────────────────────────────────────────────
        _section("FRONTEND", BLU)
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append(frontend_proc)
        threading.Thread(
            target=_stream_reader,
            args=(frontend_proc.stdout, "FRONTEND", BLU, stop, False),
            daemon=True,
        ).start()
        print(f"  {BLU}{B}[FRONTEND]{R}  pid={frontend_proc.pid}")

        # ── Wait for Vite to boot, then open browser ──────────────────
        _section("RUNNING", GRN)
        print(f"  {DIM}Waiting 2 s for Vite to start...{R}")
        time.sleep(2.0)
        webbrowser.open("http://localhost:5173")
        print(f"  {GRN}{B}✓  Browser opened → http://localhost:5173{R}")
        print(f"  {DIM}Press Ctrl+C to stop.{R}\n")

        # ── Poll for process exit ──────────────────────────────────────
        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    label = "BACKEND" if p is backend_proc else "FRONTEND"
                    col   = RED if code != 0 else YLW
                    print(f"\n  {col}{B}[{label}] exited with code {code}{R}")
                    if code != 0:
                        print(f"  {RED}Check the logs above for the error.{R}")
                    stop.set()
                    return
            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n\n  {YLW}{B}Ctrl+C — stopping Small O...{R}")

    finally:
        stop.set()
        _section("SHUTDOWN", YLW)
        for p in procs:
            label = "BACKEND" if p is backend_proc else "FRONTEND"
            _kill_tree(p, label)
        elapsed = time.perf_counter() - _SESSION_START
        m, s = divmod(int(elapsed), 60)
        print(f"\n  {DIM}Session duration: {m:02d}:{s:02d}{R}")
        print(f"  {DIM}Done.{R}\n")


if __name__ == "__main__":
    main()
