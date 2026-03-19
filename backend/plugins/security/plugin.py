import re
import subprocess
import threading

from plugins.base import BasePlugin, PluginResult


class SecurityPlugin(BasePlugin):
    NAME = "security"
    PRIORITY = 15  # Checked before internet — security patterns are specific

    INTENTS = [
        # Network info / local IP
        (r"\b(?:my\s+)?(?:local\s+)?ip\s*(?:address)?\b", "network_info"),
        (r"\bnetwork\s+(?:info|interfaces?|config|status)\b", "network_info"),
        (r"\bifconfig\b", "network_info"),
        (r"\bgateway\b", "network_info"),
        # Ping
        (r"\bping\s+(?P<host>\S+)", "ping"),
        (r"\bcheck\s+(?:if\s+)?(?P<host>\S+)\s+is\s+(?:up|alive|reachable)\b", "ping"),
        # Traceroute
        (r"\btraceroute?\s+(?P<host>\S+)", "traceroute"),
        (r"\btrace\s+(?:route\s+)?to\s+(?P<host>\S+)", "traceroute"),
        # Port scan
        (r"\bport\s+scan\b", "port_scan"),
        (r"\bscan\s+(?:open\s+)?ports?\b", "port_scan"),
        (r"\bnmap\b", "port_scan"),
        # Network discovery
        (r"\bdiscover\s+(?:devices?|hosts?|machines?)\b", "network_discovery"),
        (r"\bwho(?:'s|\s+is)\s+on\s+(?:my\s+)?(?:network|lan|wifi)\b", "network_discovery"),
        (r"\bscan\s+(?:the\s+)?network\b", "network_discovery"),
        (r"\barp\s+scan\b", "network_discovery"),
        # Wi-Fi info
        (r"\bwifi\s+(?:info|status|details|network)\b", "wifi_info"),
        (r"\bwireless\s+(?:info|network|status)\b", "wifi_info"),
        (r"\bwhat\s+(?:wifi|network)\s+am\s+i\s+(?:on|connected\s+to)\b", "wifi_info"),
        # Whois
        (r"\bwhois\s+(?P<target>\S+)", "whois_lookup"),
        (r"\bwho\s+owns\s+(?P<target>\S+)", "whois_lookup"),
        # DNS lookup
        (r"\bdns\s+lookup\s+(?P<host>\S+)", "dns_lookup"),
        (r"\bdig\s+(?P<host>\S+)", "dns_lookup"),
        (r"\bresolve\s+(?P<host>\S+)", "dns_lookup"),
        (r"\bwhat\s+(?:is\s+the\s+)?(?:ip|address)\s+(?:of|for)\s+(?P<host>\S+)", "dns_lookup"),
        # SSL cert check
        (r"\bssl\s+(?:cert(?:ificate)?|check|info)\s+(?P<host>\S+)", "ssl_check"),
        (r"\bcertificate\s+(?:info|check|expiry)\s+(?:of|for)?\s*(?P<host>\S+)", "ssl_check"),
        # HTTP headers
        (r"\bhttp\s+headers?\s+(?:of|for)?\s*(?P<url>\S+)", "http_headers"),
        (r"\bcurl\s+(?:headers?\s+)?(?:for|of)\s+(?P<url>\S+)", "http_headers"),
        (r"\bcheck\s+(?:the\s+)?headers?\s+(?:of|for)\s+(?P<url>\S+)", "http_headers"),
        # Vulnerability scan
        (r"\bvuln(?:erability)?\s+scan\b", "vuln_scan"),
        (r"\bscan\s+for\s+vuln(?:erabilities?)?\b", "vuln_scan"),
    ]

    def execute(self, user_text: str, action: str, match: re.Match) -> PluginResult:
        handler = getattr(self, f"_{action}", None)
        if handler is None:
            return {"text": f"Unknown security action: {action}", "direct": True,
                    "plugin": self.NAME, "action": action}
        try:
            return handler(user_text, match)
        except Exception as e:
            return {"text": f"Security plugin error: {e}", "direct": True,
                    "plugin": self.NAME, "action": action}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run(self, cmd: list[str], timeout: int = 30) -> str:
        """Run a command and return combined stdout/stderr output."""
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False
        )
        return (proc.stdout or proc.stderr or "No output.").strip()

    def _extract_target(self, user_text: str) -> str:
        """Try to extract a hostname/IP from the end of the user's sentence."""
        # Last token that looks like a hostname or IP
        tokens = user_text.split()
        for token in reversed(tokens):
            token = token.strip(".,?!")
            if re.match(r"^[\w.\-]+$", token) and len(token) > 2:
                return token
        return "192.168.1.0/24"  # fallback for network scans

    def _get_local_subnet(self) -> str:
        """Derive a /24 subnet from ifconfig output for network discovery."""
        try:
            out = self._run(["ifconfig"])
            ip_match = re.search(r"inet\s+(192\.168\.\d+\.\d+)", out)
            if ip_match:
                parts = ip_match.group(1).split(".")
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        except Exception:
            pass
        return "192.168.1.0/24"

    # ── Actions ───────────────────────────────────────────────────────────────

    def _network_info(self, user_text: str, match: re.Match) -> PluginResult:
        output = self._run(["ifconfig"])
        # Trim to readable IPs only
        lines = []
        for line in output.splitlines():
            if "inet " in line or "flags" in line or re.match(r"^\w", line):
                lines.append(line)
        trimmed = "\n".join(lines[:30])
        return {"text": trimmed, "direct": False, "plugin": self.NAME, "action": "network_info"}

    def _ping(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            host = match.group("host").strip(".,?!")
        except IndexError:
            host = self._extract_target(user_text)
        output = self._run(["ping", "-c", "4", host], timeout=20)
        return {"text": output, "direct": True, "plugin": self.NAME, "action": "ping"}

    def _traceroute(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            host = match.group("host").strip(".,?!")
        except IndexError:
            host = self._extract_target(user_text)
        output = self._run(["traceroute", host], timeout=60)
        return {"text": output[:800], "direct": False, "plugin": self.NAME, "action": "traceroute"}

    def _port_scan(self, user_text: str, match: re.Match) -> PluginResult:
        target = self._extract_target(user_text)
        self._run_async_scan(
            target=target,
            cmd=["nmap", "-F", "--open", target],
            action="port_scan",
            timeout=120
        )
        return {"text": f"Starting fast port scan on {target}. I'll report back when it's done.",
                "direct": True, "plugin": self.NAME, "action": "port_scan"}

    def _network_discovery(self, user_text: str, match: re.Match) -> PluginResult:
        subnet = self._get_local_subnet()
        self._run_async_scan(
            target=subnet,
            cmd=["nmap", "-sn", subnet],
            action="network_discovery",
            timeout=120
        )
        return {"text": f"Scanning {subnet} for live hosts. I'll let you know what I find.",
                "direct": True, "plugin": self.NAME, "action": "network_discovery"}

    def _wifi_info(self, user_text: str, match: re.Match) -> PluginResult:
        # Try macOS airport utility first, fall back to system_profiler
        airport = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
        try:
            output = self._run([airport, "-I"], timeout=10)
        except Exception:
            output = self._run(
                ["system_profiler", "SPAirPortDataType"],
                timeout=15
            )
        return {"text": output[:800], "direct": False, "plugin": self.NAME, "action": "wifi_info"}

    def _whois_lookup(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            target = match.group("target").strip(".,?!")
        except IndexError:
            target = self._extract_target(user_text)
        output = self._run(["whois", target], timeout=15)
        # Trim to most relevant lines (registrar, dates, status)
        relevant = []
        for line in output.splitlines():
            lower = line.lower()
            if any(k in lower for k in ["registrar", "creation", "expiry", "updated",
                                         "registrant", "country", "status", "name server"]):
                relevant.append(line.strip())
        result = "\n".join(relevant[:20]) if relevant else output[:600]
        return {"text": result, "direct": False, "plugin": self.NAME, "action": "whois_lookup"}

    def _dns_lookup(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            host = match.group("host").strip(".,?!")
        except IndexError:
            host = self._extract_target(user_text)
        output = self._run(["dig", "+short", host], timeout=10)
        if not output:
            output = f"No DNS records found for {host}."
        return {"text": output, "direct": True, "plugin": self.NAME, "action": "dns_lookup"}

    def _ssl_check(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            host = match.group("host").strip(".,?!/")
        except IndexError:
            host = self._extract_target(user_text)
        # Strip protocol if present
        host = re.sub(r"^https?://", "", host).split("/")[0]
        proc = subprocess.run(
            ["openssl", "s_client", "-connect", f"{host}:443", "-servername", host],
            input=b"Q\n",
            capture_output=True,
            timeout=15
        )
        output = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
        # Extract key fields: subject, issuer, validity dates
        relevant = []
        for line in output.splitlines():
            lower = line.lower()
            if any(k in lower for k in ["subject", "issuer", "not before", "not after",
                                         "verify return", "cn=", "o="]):
                relevant.append(line.strip())
        result = "\n".join(relevant[:15]) if relevant else output[:500]
        return {"text": result, "direct": False, "plugin": self.NAME, "action": "ssl_check"}

    def _http_headers(self, user_text: str, match: re.Match) -> PluginResult:
        try:
            url = match.group("url").strip(".,?!")
        except IndexError:
            url = self._extract_target(user_text)
        if not url.startswith("http"):
            url = "https://" + url
        output = self._run(["curl", "-I", "--max-time", "10", url], timeout=15)
        return {"text": output[:800], "direct": False,
                "plugin": self.NAME, "action": "http_headers"}

    def _vuln_scan(self, user_text: str, match: re.Match) -> PluginResult:
        target = self._extract_target(user_text)
        self._run_async_scan(
            target=target,
            cmd=["nmap", "--script", "vuln", target],
            action="vuln_scan",
            timeout=300
        )
        return {
            "text": f"Vulnerability scan started on {target}. This may take a few minutes. I'll speak the results when done.",
            "direct": True, "plugin": self.NAME, "action": "vuln_scan"
        }

    # ── Async scan helper ─────────────────────────────────────────────────────

    def _run_async_scan(self, target: str, cmd: list[str],
                        action: str, timeout: int) -> None:
        """
        Run a long nmap scan in a background daemon thread.
        When done, summarizes via LLM and speaks directly.
        This avoids blocking the main loop.
        """
        def _worker():
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=timeout, shell=False
                )
                raw = (proc.stdout or proc.stderr or "No output.").strip()
            except subprocess.TimeoutExpired:
                raw = f"Scan on {target} timed out after {timeout} seconds."
            except FileNotFoundError:
                raw = "nmap not found. Please install it with: brew install nmap"
            except Exception as e:
                raw = f"Scan failed: {e}"

            # Import here to avoid circular imports at module load time
            from llm import ask_llm
            from tts import speak
            from memory_system.core.insert_pipeline import insert_memory

            summary = ask_llm(
                f"Summarize this {action.replace('_', ' ')} result in 2-3 sentences. "
                f"Highlight open ports or interesting findings if any.\n\n{raw[:1200]}"
            )
            speak(summary)

            insert_memory({
                "text": f"Ran {action} on {target}.\nResult:\n{raw[:600]}",
                "memory_type": "ActionMemory",
                "project_reference": f"Plugin:{self.NAME}",
                "source": "plugin"
            })

        threading.Thread(target=_worker, daemon=True).start()
