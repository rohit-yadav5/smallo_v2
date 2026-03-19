import re
import subprocess
from datetime import datetime
from pathlib import Path

import psutil
import pyautogui

from plugins.base import BasePlugin, PluginResult


class ComputerPlugin(BasePlugin):
    NAME = "computer"
    PRIORITY = 10  # Checked first — computer control is unambiguous

    INTENTS = [
        # App / file / folder control
        (r"\bopen\s+(?P<target>.+)", "open_target"),
        (r"\blaunch\s+(?P<target>.+)", "open_target"),
        (r"\bclose\s+(?P<target>.+)\bapp\b", "close_app"),
        (r"\bquit\s+(?P<target>.+)\bapp\b", "close_app"),
        (r"\bkill\s+(?P<target>.+)\bapp\b", "close_app"),
        # Screenshot
        (r"\btake\s+(?:a\s+)?screenshot\b", "take_screenshot"),
        (r"\bcapture\s+(?:the\s+)?screen\b", "take_screenshot"),
        # Keyboard
        (r"\btype\s+(?P<text>.+)", "keyboard_type"),
        (r"\bpress\s+(?P<key>.+?)\s+key\b", "keyboard_press"),
        # Mouse
        (r"\bclick\b", "mouse_click"),
        (r"\bscroll\s+(?P<direction>up|down)\b", "mouse_scroll"),
        (r"\bmove\s+(?:the\s+)?mouse\b", "mouse_move"),
        # System info
        (r"\b(?:cpu|processor)\s*(?:usage|percent|load)?\b", "system_info"),
        (r"\b(?:ram|memory)\s*(?:usage|percent)?\b", "system_info"),
        (r"\bdisk\s*(?:usage|space|info)?\b", "system_info"),
        (r"\bbattery\b", "system_info"),
        (r"\bsystem\s*(?:info|stats|status)\b", "system_info"),
        # Volume
        (r"\b(?:set|turn|raise|lower|increase|decrease)\s+(?:the\s+)?volume\b", "volume_control"),
        (r"\bmute\b", "volume_control"),
        (r"\bunmute\b", "volume_control"),
        # Brightness
        (r"\b(?:set|raise|lower|increase|decrease)\s+(?:the\s+)?brightness\b", "brightness_control"),
        # Shell
        (r"\b(?:run|execute)\s+(?:the\s+)?command\s+(?P<cmd>.+)", "run_shell"),
        (r"\bin\s+(?:the\s+)?terminal\s+(?P<cmd>.+)", "run_shell"),
    ]

    def execute(self, user_text: str, action: str, match: re.Match) -> PluginResult:
        handler = getattr(self, f"_{action}", None)
        if handler is None:
            return {"text": f"Unknown computer action: {action}", "direct": True,
                    "plugin": self.NAME, "action": action}
        try:
            return handler(user_text, match)
        except Exception as e:
            return {"text": f"Computer plugin error: {e}", "direct": True,
                    "plugin": self.NAME, "action": action}

    # ── Actions ──────────────────────────────────────────────────────────────

    def _open_target(self, user_text: str, match: re.Match) -> PluginResult:
        target = match.group("target").strip()
        subprocess.run(["open", target], timeout=10)
        return {"text": f"Opening {target}.", "direct": True,
                "plugin": self.NAME, "action": "open_target"}

    def _close_app(self, user_text: str, match: re.Match) -> PluginResult:
        target = match.group("target").strip()
        script = f'tell application "{target}" to quit'
        subprocess.run(["osascript", "-e", script], timeout=10)
        return {"text": f"Closing {target}.", "direct": True,
                "plugin": self.NAME, "action": "close_app"}

    def _take_screenshot(self, user_text: str, match: re.Match) -> PluginResult:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path.home() / "Desktop" / f"screenshot_{timestamp}.png"
        subprocess.run(["screencapture", "-x", str(path)], check=True, timeout=10)
        return {"text": f"Screenshot saved to Desktop as screenshot_{timestamp}.png.",
                "direct": True, "plugin": self.NAME, "action": "take_screenshot"}

    def _keyboard_type(self, user_text: str, match: re.Match) -> PluginResult:
        text_to_type = match.group("text").strip()
        pyautogui.write(text_to_type, interval=0.05)
        return {"text": f"Typed: {text_to_type}", "direct": True,
                "plugin": self.NAME, "action": "keyboard_type"}

    def _keyboard_press(self, user_text: str, match: re.Match) -> PluginResult:
        key = match.group("key").strip().lower()
        pyautogui.press(key)
        return {"text": f"Pressed {key}.", "direct": True,
                "plugin": self.NAME, "action": "keyboard_press"}

    def _mouse_click(self, user_text: str, match: re.Match) -> PluginResult:
        pyautogui.click()
        return {"text": "Clicked.", "direct": True,
                "plugin": self.NAME, "action": "mouse_click"}

    def _mouse_scroll(self, user_text: str, match: re.Match) -> PluginResult:
        direction = match.group("direction").lower()
        clicks = 3 if direction == "up" else -3
        pyautogui.scroll(clicks)
        return {"text": f"Scrolled {direction}.", "direct": True,
                "plugin": self.NAME, "action": "mouse_scroll"}

    def _mouse_move(self, user_text: str, match: re.Match) -> PluginResult:
        # Look for coordinates like "move mouse to 500 300"
        coords = re.findall(r"\d+", user_text)
        if len(coords) >= 2:
            x, y = int(coords[0]), int(coords[1])
            pyautogui.moveTo(x, y, duration=0.3)
            return {"text": f"Moved mouse to {x}, {y}.", "direct": True,
                    "plugin": self.NAME, "action": "mouse_move"}
        return {"text": "Please specify coordinates, e.g. 'move mouse to 500 300'.",
                "direct": True, "plugin": self.NAME, "action": "mouse_move"}

    def _system_info(self, user_text: str, match: re.Match) -> PluginResult:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        battery = psutil.sensors_battery()
        bat_str = f"{battery.percent:.0f}%" if battery else "N/A"
        data = (
            f"CPU: {cpu}%\n"
            f"RAM: {mem.percent}% used ({mem.used // 1024**3} GB / {mem.total // 1024**3} GB)\n"
            f"Disk: {disk.percent}% used ({disk.used // 1024**3} GB / {disk.total // 1024**3} GB)\n"
            f"Battery: {bat_str}"
        )
        return {"text": data, "direct": False, "plugin": self.NAME, "action": "system_info"}

    def _volume_control(self, user_text: str, match: re.Match) -> PluginResult:
        lower = user_text.lower()
        level_match = re.search(r"(\d+)", user_text)
        if "unmute" in lower:
            script = "set volume output muted false"
            msg = "Unmuted."
        elif "mute" in lower:
            script = "set volume output muted true"
            msg = "Muted."
        elif level_match:
            level = max(0, min(100, int(level_match.group(1))))
            script = f"set volume output volume {level}"
            msg = f"Volume set to {level}."
        elif any(w in lower for w in ["raise", "increase", "up"]):
            script = "set volume output volume ((output volume of (get volume settings)) + 10)"
            msg = "Volume increased."
        else:
            script = "set volume output volume ((output volume of (get volume settings)) - 10)"
            msg = "Volume decreased."
        subprocess.run(["osascript", "-e", script], timeout=5)
        return {"text": msg, "direct": True, "plugin": self.NAME, "action": "volume_control"}

    def _brightness_control(self, user_text: str, match: re.Match) -> PluginResult:
        level_match = re.search(r"(\d+)", user_text)
        if not level_match:
            return {"text": "Please say a brightness level between 0 and 100.",
                    "direct": True, "plugin": self.NAME, "action": "brightness_control"}
        level = max(0, min(100, int(level_match.group(1)))) / 100.0
        script = f'tell application "System Preferences" to activate\ntell application "System Events" to set value of slider 1 of group 1 of tab group 1 of window "Displays" of application process "System Preferences" to {level}'
        subprocess.run(["osascript", "-e", script], timeout=10)
        return {"text": f"Brightness set to {int(level * 100)}%.", "direct": True,
                "plugin": self.NAME, "action": "brightness_control"}

    def _run_shell(self, user_text: str, match: re.Match) -> PluginResult:
        cmd_str = match.group("cmd").strip()
        # Split on whitespace — no shell=True, no pipes/redirects via shell
        cmd_parts = cmd_str.split()
        proc = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=30,
            shell=False
        )
        output = (proc.stdout or proc.stderr or "Command completed with no output.").strip()
        return {"text": output[:600], "direct": False,
                "plugin": self.NAME, "action": "run_shell"}
