"""backend/utils/ram_monitor.py — Runtime RAM availability monitor.

Provides lightweight helpers for checking available system RAM on Apple
Silicon Macs (and any psutil-supported platform).  Used by the planner
to select the appropriate model (7b vs 3b) at plan time, and by main.py
to log startup RAM state.

Thresholds are tuned for a 16 GB Mac running Small O with a baseline
consumption of ~12 GB (system + browsers + Small O models loaded):

  low    = > 3.0 GB free  →  safe to load qwen2.5:7b
  medium = 1.5–3.0 GB free → caution; 7b will create memory pressure
  high   = < 1.5 GB free   → danger; only use qwen2.5:3b
"""

import psutil

# ── Thresholds (GB) — change here only, referenced nowhere else ──────────────
_THRESHOLD_LOW_GB:    float = 3.0   # >= this → "low" pressure, 7b safe
_THRESHOLD_MEDIUM_GB: float = 1.5   # >= this → "medium" pressure, 7b risky


def get_available_ram_gb() -> float:
    """Return available RAM in GB (not total — what the OS considers free now)."""
    return psutil.virtual_memory().available / (1024 ** 3)


def get_memory_pressure() -> str:
    """
    Return 'low', 'medium', or 'high' based on available RAM right now.

    Thresholds are defined as module-level constants above.
    """
    available = get_available_ram_gb()
    if available < _THRESHOLD_MEDIUM_GB:
        return "high"
    elif available < _THRESHOLD_LOW_GB:
        return "medium"
    else:
        return "low"


def can_load_7b() -> bool:
    """
    Return True if it's safe to load qwen2.5:7b right now.

    Requires at least _THRESHOLD_LOW_GB (3.0 GB) free to avoid swap
    pressure on a 16 GB Apple Silicon machine.
    """
    return get_available_ram_gb() >= _THRESHOLD_LOW_GB
