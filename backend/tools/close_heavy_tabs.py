"""backend/tools/close_heavy_tabs.py — Close memory-heavy browser tabs.

Closes tabs in Chrome and Safari matching a list of known heavy domains
(YouTube, Suno, ChatGPT, etc.) using macOS AppleScript via osascript.

macOS only — returns an error string on non-macOS platforms.

RAM estimate: each closed tab frees ~400 MB on average (typical for
GPU-accelerated video/audio sites with large JS heaps).
"""

import platform
import re
import subprocess

from tools.registry import registry, ToolDefinition


# Domains whose tabs are considered "heavy" and eligible for closure.
# Applied as substring matches against the tab URL.
_HEAVY_DOMAINS = [
    "youtube.com",
    "suno.com",
    "chatgpt.com",
    "netflix.com",
    "twitch.tv",
    "spotify.com",
]

# Average RAM freed per closed tab (MB) — used for the estimate message.
_MB_PER_TAB = 400

# AppleScript template — {domains_predicate} is expanded at runtime.
_CHROME_SCRIPT = """\
tell application "Google Chrome"
    if not running then return "Chrome not running"
    set closedCount to 0
    repeat with w in every window
        set tabsToClose to {}
        repeat with t in every tab of w
            set u to URL of t
            if {domain_checks} then
                set end of tabsToClose to t
            end if
        end repeat
        repeat with t in tabsToClose
            close t
            set closedCount to closedCount + 1
        end repeat
    end repeat
    return closedCount as text
end tell
"""

_SAFARI_SCRIPT = """\
tell application "Safari"
    if not running then return "Safari not running"
    set closedCount to 0
    repeat with w in every window
        set tabsToClose to {}
        repeat with t in every tab of w
            set u to URL of t
            if u is not missing value and ({domain_checks}) then
                set end of tabsToClose to t
            end if
        end repeat
        repeat with t in tabsToClose
            close t
            set closedCount to closedCount + 1
        end repeat
    end repeat
    return closedCount as text
end tell
"""


def _build_domain_checks(url_var: str = "u") -> str:
    """Build the AppleScript domain predicate for the given URL variable."""
    parts = [f'{url_var} contains "{d}"' for d in _HEAVY_DOMAINS]
    return " or ".join(parts)


def _run_osascript(script: str) -> tuple[int, str]:
    """Run an AppleScript string via osascript and return (return_code, output)."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode, (result.stdout.strip() or result.stderr.strip())
    except subprocess.TimeoutExpired:
        return 1, "osascript timed out"
    except Exception as exc:
        return 1, str(exc)


async def _close_heavy_tabs(args: dict) -> str:
    """Close heavy browser tabs in Chrome + Safari using AppleScript."""
    if platform.system() != "Darwin":
        return "close_heavy_tabs is only supported on macOS."

    domain_checks = _build_domain_checks("u")

    chrome_script = _CHROME_SCRIPT.replace("{domain_checks}", domain_checks)
    safari_script  = _SAFARI_SCRIPT.replace("{domain_checks}", domain_checks)

    chrome_rc, chrome_out = _run_osascript(chrome_script)
    safari_rc,  safari_out = _run_osascript(safari_script)

    total_closed = 0
    notes: list[str] = []

    # Parse Chrome result
    if chrome_rc == 0 and chrome_out.isdigit():
        total_closed += int(chrome_out)
    elif "not running" in chrome_out:
        notes.append("Chrome not running")
    elif chrome_rc != 0:
        notes.append(f"Chrome error: {chrome_out[:80]}")

    # Parse Safari result
    if safari_rc == 0 and safari_out.isdigit():
        total_closed += int(safari_out)
    elif "not running" in safari_out:
        notes.append("Safari not running")
    elif safari_rc != 0:
        notes.append(f"Safari error: {safari_out[:80]}")

    freed_mb = total_closed * _MB_PER_TAB
    summary = f"Closed {total_closed} heavy tab{'s' if total_closed != 1 else ''}."
    if freed_mb > 0:
        summary += f" Freed approximately {freed_mb:,} MB."
    if notes:
        summary += f" ({'; '.join(notes)})"

    return summary


# ── Self-registration ──────────────────────────────────────────────────────────

registry.register(ToolDefinition(
    name="close_heavy_tabs",
    description=(
        "Close memory-heavy browser tabs (YouTube, Suno, ChatGPT, Netflix, Twitch, Spotify) "
        "in Chrome and Safari to free RAM before intensive tasks. macOS only."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    handler=_close_heavy_tabs,
))
