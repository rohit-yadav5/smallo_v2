"""backend/planner — Autonomous multi-step task planning for Small O.

Phase 2 of the Jarvis upgrade.  The planner receives a one-sentence goal,
decomposes it into concrete steps, executes each step using the ToolRegistry,
and summarises what was accomplished — all without asking the user for
permission at each stage.

Public API
──────────
  from planner.planner import run_plan

  await run_plan(goal="…", broadcast=_emit, max_steps=20)

The ``broadcast`` argument must match the signature of ``_emit`` in
backend/main.py:  broadcast(event: str, data: dict) -> None
"""

from planner.planner import run_plan  # noqa: F401
