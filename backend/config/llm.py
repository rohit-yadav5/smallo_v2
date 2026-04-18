"""backend/config/llm.py — LLM configuration for Small O.

Centralises every Ollama-related constant behind a frozen dataclass so that
backend/llm/main_llm.py and any future sub-agent callers share a single source
of truth.  All values can be overridden with environment variables — useful for
running different models on different machines without editing code.

Two-model split
───────────────
  model          — qwen2.5:3b  — fast, used for conversational turns only
  planner_model  — qwen2.5:7b  — higher-capacity, used for plan decomposition,
                                  per-step execution, goal checks, and summaries

This keeps voice interaction latency low while giving the planner the reasoning
capacity needed to avoid URL hallucination and goal drift.
"""

import os
from dataclasses import dataclass

# ── Three-tier keep_alive strategy ───────────────────────────────────────────
# Controls how long Ollama keeps the model in VRAM after a call completes.
# Named constants here so every caller (main_llm, planner) stays in sync.
#   IDLE   — no active conversation: evict immediately to free ~1.2 GB RAM
#   ACTIVE — conversation in progress: stay warm between turns (120 s window)
#   PLAN   — plan execution: never evict mid-plan (10 min window)
KEEP_ALIVE_IDLE   = "0s"    # evict after call — RAM freed within seconds
KEEP_ALIVE_ACTIVE = "120s"  # stay warm during active conversation
KEEP_ALIVE_PLAN   = "600s"  # stay warm for full plan execution (10 min)


@dataclass(frozen=True)
class LLMConfig:
    ollama_url:               str
    model:                    str    # conversational turns — qwen2.5:3b (fast)
    planner_model:            str    # planner all phases  — qwen2.5:7b (accurate)
    num_predict:              int
    planner_num_predict:      int    # larger budget for multi-step reasoning
    stream_timeout_connect_s: float
    stream_timeout_read_s:    float


LLM_CONFIG = LLMConfig(
    ollama_url               = os.getenv("OLLAMA_URL",          "http://localhost:11434/api/chat"),
    model                    = os.getenv("LLM_MODEL",           "qwen2.5:7b"),
    planner_model            = os.getenv("PLANNER_MODEL",       "qwen2.5:7b"),
    num_predict              = int(os.getenv("LLM_NUM_PREDICT",         "512")),
    planner_num_predict      = int(os.getenv("PLANNER_NUM_PREDICT",    "1024")),
    stream_timeout_connect_s = float(os.getenv("LLM_TIMEOUT_CONNECT",    "10")),
    stream_timeout_read_s    = float(os.getenv("LLM_TIMEOUT_READ",      "120")),
)

# ── Memory system constants ───────────────────────────────────────────────────
# Importance decay half-lives in days (None = never decays).
DECAY_HALF_LIFE: dict[str, int | None] = {
    "PersonalMemory":    180,
    "DecisionMemory":     90,
    "ActionMemory":       30,
    "PlannerMemory":      14,
    "ConsolidatedMemory": None,
}
DECAY_HALF_LIFE_DEFAULT = 60  # fallback for unlisted memory types

# Consolidation thresholds
CONSOLIDATION_SIMILARITY_THRESHOLD    = 0.75   # min similarity to cluster memories
META_CONSOLIDATION_SIMILARITY_THRESHOLD = 0.80  # min similarity to merge ConsolidatedMemorys
CONSOLIDATION_VALIDATION_THRESHOLD    = 0.60   # min sim between LLM summary and cluster centroid
CONSOLIDATED_MEMORY_EXPIRY_DAYS       = 90
CONSOLIDATED_MEMORY_EXPIRY_MIN_IMPORTANCE = 6.0
