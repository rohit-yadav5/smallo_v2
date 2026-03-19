import time
from contextlib import contextmanager


class LatencyTracker:
    """
    Per-turn latency tracker.

    Usage:
        tracker = LatencyTracker()
        with tracker.step("STT"):
            ...
        tracker.summary()
    """

    _BAR_WIDTH = 18

    def __init__(self):
        self._turn_start = time.perf_counter()
        self._steps: list[tuple[str, float]] = []

    @contextmanager
    def step(self, name: str):
        print(f"  → {name}", flush=True)
        t0 = time.perf_counter()
        yield
        duration = time.perf_counter() - t0
        cumulative = time.perf_counter() - self._turn_start
        self._steps.append((name, duration))
        print(f"  ✓ {name:<32} {duration:6.3f}s   [+{cumulative:.3f}s]")

    def summary(self):
        total = time.perf_counter() - self._turn_start
        longest = max((d for _, d in self._steps), default=1.0)

        print()
        print("  ┌─ Turn Breakdown " + "─" * 35 + "┐")
        for name, duration in self._steps:
            pct = (duration / total * 100) if total > 0 else 0
            bar_len = int((duration / longest) * self._BAR_WIDTH)
            bar = "█" * bar_len
            print(f"  │  {name:<28}  {duration:6.3f}s  {pct:5.1f}%  {bar}")
        print("  ├" + "─" * 52 + "┤")
        print(f"  │  {'TOTAL':<28}  {total:6.3f}s")
        print("  └" + "─" * 52 + "┘")
        print()
