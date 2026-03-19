"""main.py – Start Small O: backend WebSocket server + frontend dev server."""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"


def main():
    procs = []

    try:
        # ── Frontend (Vite dev server) ────────────────────────────────────────
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND,
        )
        procs.append(frontend_proc)
        print("  [frontend] started → http://localhost:5173")

        # ── Backend (WebSocket server) ────────────────────────────────────────
        backend_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=BACKEND,
        )
        procs.append(backend_proc)

        # Wait for either process to exit
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"\n  Process exited with code {p.returncode}. Shutting down.")
                    return

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


if __name__ == "__main__":
    main()
