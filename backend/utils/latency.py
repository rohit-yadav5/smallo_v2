import time
from contextlib import contextmanager
from datetime import datetime

# ── ANSI colours ─────────────────────────────────────────────────────
_R  = "\033[0m"
_B  = "\033[1m"
_DIM= "\033[2m"
_GRN= "\033[92m"
_YLW= "\033[93m"
_RED= "\033[91m"
_CYN= "\033[96m"
_MAG= "\033[95m"
_WHT= "\033[97m"
_BLU= "\033[94m"

# Speed thresholds (seconds)
_FAST   = 0.5
_MEDIUM = 2.0

def _speed_color(secs: float) -> str:
    if secs < _FAST:
        return _GRN
    if secs < _MEDIUM:
        return _YLW
    return _RED

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:12]

def _bar(duration: float, longest: float, width: int = 20) -> str:
    if longest <= 0:
        return ""
    filled = int((duration / longest) * width)
    return "█" * filled + "░" * (width - filled)


class LatencyTracker:
    """
    Per-turn latency tracker with detailed timing, colour coding,
    bottleneck detection, and sub-step annotations.

    Usage:
        tracker = LatencyTracker(turn=1)
        with tracker.step("STT"):
            ...
            tracker.note("audio: 2.1s, 33 KB")
        tracker.summary()
    """

    _BAR_WIDTH = 20

    def __init__(self, turn: int = 0):
        self._turn       = turn
        self._turn_start = time.perf_counter()
        self._steps      : list[tuple[str, float, list[str]]] = []
        self._current_notes: list[str] = []
        self._header_lines : list[str] = []   # printed before steps in summary
        self._cancel_step  : bool      = False # set inside step() to skip recording

    def note(self, msg: str):
        """Attach a detail annotation to the current step."""
        self._current_notes.append(msg)
        print(f"  {_DIM}       ↳ {msg}{_R}", flush=True)

    def header(self, msg: str):
        """
        Record a turn-level note shown at the top of the summary box.
        Use this for metadata like 'source: text input' that isn't a timed step.
        """
        self._header_lines.append(msg)

    def cancel_current_step(self):
        """
        Cancel the currently-running step() so it is NOT appended to the
        step list and does NOT appear in summary().  Must be called from
        inside an active `with tracker.step(...)` block.
        """
        self._cancel_step = True

    def record(self, name: str, duration: float, notes: list[str] | None = None):
        """Record a pre-measured step (timing captured externally).

        Use this when the code being timed runs outside a context manager —
        e.g. when a single function (speak_stream) internally measures multiple
        sub-phases and returns them in a timing dict.
        """
        ts         = _ts()
        cumulative = time.perf_counter() - self._turn_start
        col        = _speed_color(duration)
        notes      = notes or []
        print(f"\n  {_BLU}{_B}▶ {name}{_R}  {_DIM}({ts}){_R}", flush=True)
        for n in notes:
            print(f"  {_DIM}       ↳ {n}{_R}", flush=True)
        print(
            f"  {col}{_B}✓ {name}{_R}"
            f"  {col}{duration:.3f}s{_R}"
            f"  {_DIM}[total {cumulative:.3f}s]{_R}",
            flush=True,
        )
        self._steps.append((name, duration, notes))

    @contextmanager
    def step(self, name: str):
        ts = _ts()
        print(f"\n  {_BLU}{_B}▶ {name}{_R}  {_DIM}({ts}){_R}", flush=True)
        t0 = time.perf_counter()
        self._current_notes = []
        self._cancel_step   = False   # reset for each new step
        try:
            yield
        except Exception as exc:
            duration = time.perf_counter() - t0
            if not self._cancel_step:
                self._steps.append((name, duration, list(self._current_notes)))
            print(f"  {_RED}{_B}✗ {name}{_R}  {_RED}{duration:.3f}s  ERROR: {exc}{_R}", flush=True)
            raise
        duration   = time.perf_counter() - t0
        cumulative = time.perf_counter() - self._turn_start
        if self._cancel_step:
            # Step was cancelled (e.g. text input arrived during voice wait).
            # Don't record in _steps; print a dim cancellation notice instead.
            print(
                f"  {_DIM}[{name} step cancelled — not counted in summary]{_R}",
                flush=True,
            )
            return
        col = _speed_color(duration)
        self._steps.append((name, duration, list(self._current_notes)))
        print(
            f"  {col}{_B}✓ {name}{_R}"
            f"  {col}{duration:.3f}s{_R}"
            f"  {_DIM}[total {cumulative:.3f}s]{_R}",
            flush=True,
        )

    def summary(self):
        total   = time.perf_counter() - self._turn_start
        longest = max((d for _, d, _ in self._steps), default=1.0)

        # Find bottleneck
        if self._steps:
            bottleneck_name, bottleneck_dur, _ = max(self._steps, key=lambda x: x[1])
        else:
            bottleneck_name, bottleneck_dur = "–", 0.0

        print()
        print(f"  {_WHT}{_B}┌─ Turn {self._turn} Breakdown {'─' * 38}┐{_R}")
        for line in self._header_lines:
            print(f"  {_WHT}│{_R}  {_CYN}{line}{_R}")
        for name, dur, notes in self._steps:
            col    = _speed_color(dur)
            pct    = (dur / total * 100) if total > 0 else 0
            bar    = _bar(dur, longest, self._BAR_WIDTH)
            flag   = f"  {_RED}◀ BOTTLENECK{_R}" if name == bottleneck_name else ""
            print(
                f"  {_WHT}│{_R}  {name:<30}"
                f"  {col}{dur:6.3f}s{_R}"
                f"  {_DIM}{pct:5.1f}%{_R}"
                f"  {col}{bar}{_R}"
                f"{flag}"
            )
            for n in notes:
                print(f"  {_WHT}│{_R}     {_DIM}↳ {n}{_R}")

        print(f"  {_WHT}├{'─' * 64}┤{_R}")
        total_col = _speed_color(total / max(len(self._steps), 1))
        print(
            f"  {_WHT}│{_R}  {'TOTAL':<30}  {total_col}{_B}{total:6.3f}s{_R}"
            f"  {_DIM}({len(self._steps)} steps){_R}"
        )
        print(f"  {_WHT}│{_R}  {'BOTTLENECK':<30}  {_YLW}{bottleneck_name}  {bottleneck_dur:.3f}s{_R}")
        print(f"  {_WHT}└{'─' * 64}┘{_R}")
        print()
