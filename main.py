"""main.py – Start Small O: backend WebSocket server + frontend dev server.

Usage:
    python main.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
BACKEND  = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV_PY  = ROOT / ".venv" / "bin" / "python3"

# Use the venv Python if it exists, otherwise fall back to the current interpreter.
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable


def main():
    env = os.environ.copy()
    # Ensure backend modules are importable when running backend/main.py
    env["PYTHONPATH"] = str(BACKEND)

    procs: list[subprocess.Popen] = []

    try:
        # ── Frontend (Vite dev server) ────────────────────────────────────────
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND,
            env=env,
        )
        procs.append(frontend_proc)
        print("  [frontend] Vite started  →  http://localhost:5173")

        # ── Backend (WebSocket server) ────────────────────────────────────────
        backend_proc = subprocess.Popen(
            [PYTHON, "main.py"],
            cwd=BACKEND,
            env=env,
        )
        procs.append(backend_proc)
        print(f"  [backend]  WebSocket started  →  ws://localhost:8765  (python: {PYTHON})")

        # Poll until one process exits
        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    label = "frontend" if p is frontend_proc else "backend"
                    print(f"\n  [{label}] exited with code {code}. Shutting everything down.")
                    return
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n  Ctrl+C — stopping Small O...")

    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("  All processes stopped.")


if __name__ == "__main__":
    main()
