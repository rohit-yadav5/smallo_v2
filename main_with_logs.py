"""main_with_logs.py – Small O with full diagnostic logging.

Streams timestamped, colour-coded stdout+stderr from both the backend
pipeline and the Vite frontend.  Use this when debugging audio/connection
issues instead of the silent main.py.

Usage:
    python main_with_logs.py
"""
import os
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

# ── ANSI ─────────────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
CYN = "\033[96m"
MAG = "\033[95m"
BLU = "\033[94m"
WHT = "\033[97m"

# ── Helpers ───────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:12]

def _banner():
    print()
    print(f"  {WHT}{B}╔══════════════════════════════════════════════════╗{R}")
    print(f"  {WHT}{B}║          Small O  —  Diagnostic Mode            ║{R}")
    print(f"  {WHT}{B}╚══════════════════════════════════════════════════╝{R}")
    print()
    print(f"  {DIM}Python   : {PYTHON}{R}")
    print(f"  {DIM}Backend  : {BACKEND}{R}")
    print(f"  {DIM}Frontend : {FRONTEND}{R}")
    print()

def _section(title: str, color: str = WHT):
    w = 50
    pad = (w - len(title) - 2) // 2
    print(f"\n  {color}{B}{'─' * pad} {title} {'─' * pad}{R}\n")


# ── Stream readers ────────────────────────────────────────────────────
def _stream_reader(pipe, label: str, col: str, stop_event: threading.Event):
    """Read lines from a subprocess pipe and print with colour-coded prefix."""
    prefix = f"{col}{B}[{label}]{R} "
    try:
        for raw in pipe:
            if stop_event.is_set():
                break
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            # Colour-code certain keywords
            if any(k in line.lower() for k in ("error", "exception", "traceback", "failed", "crash")):
                line_col = RED
            elif any(k in line.lower() for k in ("warn", "warning")):
                line_col = YLW
            elif any(k in line.lower() for k in ("ready", "ok", "✓", "connected", "started")):
                line_col = GRN
            elif any(k in line.lower() for k in ("▶", "→", "listening", "audio", "stt", "llm", "tts")):
                line_col = CYN
            else:
                line_col = ""
            ts = _ts()
            print(f"  {DIM}{ts}{R}  {prefix}{line_col}{line}{R}", flush=True)
    except Exception:
        pass


# ── Health checks ─────────────────────────────────────────────────────
def _check_port_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0

def _preflight():
    ok = True
    print(f"  {B}Pre-flight checks{R}")

    # Python
    py_ok = Path(PYTHON).exists()
    sym = f"{GRN}✓{R}" if py_ok else f"{RED}✗{R}"
    print(f"    {sym}  Python executable  {DIM}{PYTHON}{R}")
    if not py_ok:
        print(f"       {RED}→ .venv not found — run:  python -m venv .venv && pip install -r requirements.txt{R}")
        ok = False

    # Node modules
    nm = FRONTEND / "node_modules"
    nm_ok = nm.exists()
    sym = f"{GRN}✓{R}" if nm_ok else f"{RED}✗{R}"
    print(f"    {sym}  node_modules  {DIM}{nm}{R}")
    if not nm_ok:
        print(f"       {RED}→ run:  cd frontend && npm install{R}")
        ok = False

    # Ports free
    for port, name in [(8765, "WebSocket :8765"), (5173, "Vite :5173")]:
        free = _check_port_free(port)
        sym  = f"{GRN}✓{R}" if free else f"{YLW}!{R}"
        note = "" if free else f"  {YLW}← port busy, may conflict{R}"
        print(f"    {sym}  {name} free{note}")

    print()
    return ok


# ── Main ──────────────────────────────────────────────────────────────
def main():
    _banner()
    if not _preflight():
        print(f"  {RED}Pre-flight failed — fix the issues above and retry.{R}\n")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND)
    env["FORCE_COLOR"] = "1"   # keep ANSI in subprocess output
    env["PYTHONUTF8"] = "1"

    procs: list[subprocess.Popen] = []
    stop  = threading.Event()

    try:
        # ── Backend ──────────────────────────────────────────────────
        _section("BACKEND", MAG)
        backend_proc = subprocess.Popen(
            [PYTHON, "main.py"],
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
            args=(backend_proc.stdout, "BACKEND", MAG, stop),
            daemon=True,
        ).start()
        print(f"  {MAG}{B}[BACKEND]{R}  pid={backend_proc.pid}  python={PYTHON}")

        # ── Frontend ─────────────────────────────────────────────────
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
            args=(frontend_proc.stdout, "FRONTEND", BLU, stop),
            daemon=True,
        ).start()
        print(f"  {BLU}{B}[FRONTEND]{R}  pid={frontend_proc.pid}")

        # ── Wait for Vite to boot, then open browser ──────────────────
        _section("RUNNING", GRN)
        print(f"  {DIM}Waiting 2s for Vite to start...{R}")
        time.sleep(2.0)
        webbrowser.open("http://localhost:5173")
        print(f"  {GRN}{B}✓ Browser opened → http://localhost:5173{R}")
        print(f"  {DIM}Press Ctrl+C to stop.{R}\n")

        # ── Poll for process exit ─────────────────────────────────────
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
            try:
                p.terminate()
                p.wait(timeout=5)
                print(f"  {GRN}✓{R}  {label} stopped")
            except subprocess.TimeoutExpired:
                p.kill()
                print(f"  {YLW}!{R}  {label} killed (didn't stop in time)")
            except Exception:
                pass
        print(f"\n  {DIM}Done.{R}\n")


if __name__ == "__main__":
    main()
