"""backend/config/limits.py — Centralised numeric limits and thresholds.

Import these constants instead of hardcoding magic numbers throughout the
backend.  All values here were previously scattered across multiple modules.
"""

# ── Planner ───────────────────────────────────────────────────────────────────
PLAN_TIMEOUT_S          = 45       # per-step LLM call timeout (seconds)
PLAN_MAX_STEPS          = 20       # hard cap on plan steps

# ── Conversation ──────────────────────────────────────────────────────────────
CONVERSATION_IDLE_S     = 90       # seconds before keep_alive drops to idle
LLM_TOKEN_TIMEOUT_S     = 120      # seconds with no tokens → abandon call
PARTIAL_RESPONSE_CHARS  = 400      # max barge-in context snippet chars

# ── Model warmup ──────────────────────────────────────────────────────────────
MODEL_PRELOAD_WARM_S    = 120      # seconds to keep model warm after last use
STT_WARMUP_TIMEOUT_S    = 30       # max seconds for STT JIT warmup

# ── Tool output truncation (chars) ───────────────────────────────────────────
TOOL_OUTPUT_BROWSER     = 8_000
TOOL_OUTPUT_FILE        = 50_000
TOOL_OUTPUT_TERMINAL    = 6_000

# ── Browser ───────────────────────────────────────────────────────────────────
BROWSER_WAIT_MS         = 2_000    # networkidle wait timeout for screenshots

# ── Memory deduplication ─────────────────────────────────────────────────────
DEDUP_THRESHOLD         = 0.90     # cosine similarity → exact duplicate
NEAR_DEDUP_THRESHOLD    = 0.75     # cosine similarity → near-duplicate chain

# ── STT ────────────────────────────────────────────────────────────────────────
VAD_NO_SPEECH_THRESHOLD = 0.65     # partial transcription no-speech gate
