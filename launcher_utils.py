"""launcher_utils.py — Shared helpers for main.py and main_with_logs.py."""
import subprocess
from pathlib import Path


def find_npm() -> str:
    """Return the absolute path to npm, searching common install locations."""
    import shutil
    npm = shutil.which("npm")
    if npm:
        return npm
    candidates = [
        "/opt/homebrew/bin/npm",
        "/usr/local/bin/npm",
        "/usr/bin/npm",
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    nvm_dir = Path.home() / ".nvm" / "versions" / "node"
    if nvm_dir.is_dir():
        for v in sorted(nvm_dir.iterdir(), reverse=True):
            npm_bin = v / "bin" / "npm"
            if npm_bin.is_file():
                return str(npm_bin)
    return "npm"


def kill_tree(proc: subprocess.Popen, label: str) -> None:
    """Terminate a process and all its children."""
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


def kill_port(port: int) -> bool:
    """
    Kill any process listening on the given port.

    Returns True if a process was found and killed, False if port was already free.
    Uses lsof on macOS/Linux. Safe to call even if port is already free.
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        if not pids:
            return False
        for pid in pids:
            if pid.isdigit():
                subprocess.run(["kill", "-9", pid], capture_output=True)
                print(f"  !  killed stale process on port {port} (pid={pid})")
        return True
    except Exception as exc:
        print(f"  !  could not kill port {port}: {exc}")
        return False


def check_port_free(port: int) -> bool:
    """Return True if port is not in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0
