"""backend/planner/validator.py — Post-decomposition step validation.

Runs immediately after the LLM produces a decomposed step list.
Filters and repairs bad steps before the execution loop begins.

Three failure modes caught here:
  1. Steps too vague to execute (< 4 words)
  2. System-prompt text leaked into step content (prompt echo)
  3. Hallucinated URLs with fake/unknown TLDs
  4. Steps completely unrelated to the goal (off-topic drift)

If the validator rejects ALL steps it returns the originals with a warning
so the plan can still attempt execution rather than silently failing.
"""

import re


# TLDs that Qwen2.5:3b commonly invents — none of these are real.
KNOWN_FAKE_TLDS = {'.yawq', '.fakedomain', '.notreal', '.example', '.test', '.localhost'}

# Strings that appear in system-prompt instructions but should never surface
# in a plain-English step description.  Their presence means the model echoed
# part of its prompt rather than generating a real step.
_ARTIFACTS = [
    "wrong formats",
    "never use these",
    "tool_call",
    "start_plan",
    "you are a",
    "your job is",
    "rules:",
    "respond with",
    "example:",
    "available tools",
    "output rules",
    "absolute rules",
]

# Words that count as "action words" for the relevance check — a step is
# considered on-topic even if no goal-word appears, provided it's clearly
# an action step (navigate, search, read, …).
_ACTION_WORDS = {
    'navigate', 'go', 'open', 'search', 'find', 'read', 'write', 'create',
    'click', 'type', 'get', 'fetch', 'visit', 'check', 'look', 'show',
    'list', 'make', 'download', 'extract', 'summarize', 'summarise',
}


def _log(msg: str) -> None:
    # Single-line log matching the [planner] prefix style used elsewhere.
    print(f"  {msg}", flush=True)


def validate_steps(steps: list[str], goal: str) -> list[str]:
    """
    Validate and filter a decomposed step list before execution.

    Parameters
    ----------
    steps:  Raw numbered-list entries from the decomposer.
    goal:   Original goal string — used for relevance checking.

    Returns
    -------
    Cleaned list.  If every step is rejected, returns the original list
    with a warning so execution can still proceed.
    """
    clean: list[str] = []
    goal_words = set(goal.lower().split())

    for step in steps:
        s = step.strip()
        if not s:
            continue

        # ── 1. Too short ───────────────────────────────────────────────────
        if len(s.split()) < 4:
            _log(f"[validator] rejected (too short): {s!r}")
            continue

        # ── 2. Prompt leak / artifact ──────────────────────────────────────
        s_lower = s.lower()
        if any(artifact in s_lower for artifact in _ARTIFACTS):
            _log(f"[validator] rejected (prompt leak): {s!r}")
            continue

        # ── 3. Hallucinated URL detection ──────────────────────────────────
        urls = re.findall(r'https?://\S+', s)
        rejected_for_url = False
        for url in urls:
            m = re.search(r'https?://([^/\s]+)', url)
            if m:
                domain = m.group(1)
                tld = ('.' + domain.split('.')[-1]) if '.' in domain else ''
                if tld.lower() in KNOWN_FAKE_TLDS:
                    _log(f"[validator] rejected (fake URL {url!r}): {s!r}")
                    rejected_for_url = True
                    break
        if rejected_for_url:
            continue

        # ── 4. Goal relevance ──────────────────────────────────────────────
        # Only applied when the goal is descriptive enough (≥ 4 words).
        # A step passes if:
        #   a) at least one goal word appears in the step, OR
        #   b) the step contains a recognised action word (it's a generic
        #      navigation/reading step that serves any goal)
        if len(goal_words) >= 4:
            step_words = set(s_lower.split())
            has_goal_word   = bool(goal_words & step_words)
            has_action_word = bool(step_words & _ACTION_WORDS)
            if not has_goal_word and not has_action_word:
                _log(f"[validator] rejected (off-topic): {s!r}")
                continue

        clean.append(s)

    # Safety net: if everything was rejected, fall back to originals.
    if not clean:
        _log("[validator] WARNING: all steps rejected — falling back to originals")
        return steps

    return clean
