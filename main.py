"""main.py – Start Small O: backend WebSocket server + frontend dev server.

Pipeline: Browser mic → AudioWorklet → WebSocket → Silero VAD → Whisper → LLM → Piper TTS
Barge-in: user speech during TTS cuts playback immediately; partial response is saved and
          injected as context into the next LLM call so the bot knows where it left off.

Usage:
    python main.py
"""
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
BACKEND  = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV_PY  = ROOT / ".venv" / "bin" / "python3"
PYTHON   = str(VENV_PY) if VENV_PY.exists() else sys.executable

from launcher_utils import find_npm as _find_npm_fn, kill_tree, kill_port, check_port_free

NPM = _find_npm_fn()


def main():
    env = os.environ.copy()
    env["PYTHONPATH"]       = str(BACKEND)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"]       = "1"

    procs: list[subprocess.Popen] = []
    stop          = threading.Event()
    frontend_proc: subprocess.Popen | None = None
    backend_proc:  subprocess.Popen | None = None

    try:
        # Kill any stale process holding port 8765 so re-runs don't fail
        if not check_port_free(8765):
            print("  !  port 8765 busy — killing stale process...")
            kill_port(8765)
            time.sleep(0.5)

        backend_proc = subprocess.Popen(
            [PYTHON, "-u", "main.py"],   # -u: unbuffered stdout/stderr
            cwd=BACKEND, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(backend_proc)

        frontend_proc = subprocess.Popen(
            [NPM, "run", "dev"],
            cwd=FRONTEND, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        procs.append(frontend_proc)

        # Parse the actual port Vite chose — it auto-increments when 5173 is busy.
        _vite_url  = ["http://localhost:5173"]   # list so the thread can write to it
        _url_ready = threading.Event()

        def _watch_vite_stdout():
            try:
                for line in frontend_proc.stdout:
                    m = re.search(r"Local:\s*(http://localhost:\d+)", line)
                    if m:
                        _vite_url[0] = m.group(1).rstrip("/")
                        _url_ready.set()
                        break
            except Exception:
                pass
            finally:
                _url_ready.set()   # unblock on EOF or error too

        threading.Thread(target=_watch_vite_stdout, daemon=True).start()
        _url_ready.wait(timeout=15.0)

        webbrowser.open(_vite_url[0])
        print(f"  Small O is running  →  {_vite_url[0]}")
        print("  Press Ctrl+C to stop.\n")

        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    label = "BACKEND" if p is backend_proc else "FRONTEND"
                    print(f"\n  [{label}] exited with code {code}. Shutting down.")
                    if code != 0:
                        print(f"  Run python main_with_logs.py for detailed diagnostics.")
                    stop.set()
                    return
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n  Stopping Small O...")

    finally:
        stop.set()
        for p, label in [(frontend_proc, "frontend"), (backend_proc, "backend")]:
            if p is not None:
                kill_tree(p, label)
        print("  Done.")


if __name__ == "__main__":
    main()
