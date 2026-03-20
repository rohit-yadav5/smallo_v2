"""main.py – Start Small O: backend WebSocket server + frontend dev server.

Usage:
    python main.py
"""
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
BACKEND  = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV_PY  = ROOT / ".venv" / "bin" / "python3"
DEVNULL  = subprocess.DEVNULL

# Use the venv Python if it exists, otherwise fall back to the current interpreter.
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND)

    procs: list[subprocess.Popen] = []

    try:
        # ── Backend — fully silent ────────────────────────────────────────────
        backend_proc = subprocess.Popen(
            [PYTHON, "main.py"],
            cwd=BACKEND,
            env=env,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        procs.append(backend_proc)

        # ── Frontend — fully silent ───────────────────────────────────────────
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND,
            env=env,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        procs.append(frontend_proc)

        # Give Vite ~1.5 s to start, then open the browser automatically
        time.sleep(1.5)
        webbrowser.open("http://localhost:5173")
        print("  Small O is running  →  http://localhost:5173")
        print("  Press Ctrl+C to stop.\n")

        # Poll until one process exits
        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    label = "frontend" if p is frontend_proc else "backend"
                    print(f"\n  [{label}] crashed (code {code}). Shutting down.")
                    return
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n  Stopping Small O...")

    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("  Done.")


if __name__ == "__main__":
    main()
