"""main.py – Start Small O: backend WebSocket server + frontend dev server.

Pipeline: Browser mic → AudioWorklet → WebSocket → Silero VAD → Whisper → LLM → Piper TTS
Barge-in: user speech during TTS cuts playback immediately; partial response is saved and
          injected as context into the next LLM call so the bot knows where it left off.

Usage:
    python main.py
"""
import os
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


def _kill_tree(proc: subprocess.Popen, label: str) -> None:
    """Terminate a process and all its children (handles npm → vite → esbuild chains)."""
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
        print(f"  ✓  {label} stopped")
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=5)
            print(f"  ✓  {label} stopped")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"  !  {label} killed (didn't stop in time)")
        except Exception:
            pass


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
        backend_proc = subprocess.Popen(
            [PYTHON, "-u", "main.py"],   # -u: unbuffered stdout/stderr
            cwd=BACKEND, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(backend_proc)

        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(frontend_proc)

        time.sleep(2.0)
        webbrowser.open("http://localhost:5173")
        print("  Small O is running  →  http://localhost:5173")
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
                _kill_tree(p, label)
        print("  Done.")


if __name__ == "__main__":
    main()
