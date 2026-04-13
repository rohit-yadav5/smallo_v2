"""backend/config/llm.py — LLM configuration for Small O.

Centralises every Ollama-related constant behind a frozen dataclass so that
backend/llm/main_llm.py and any future sub-agent callers share a single source
of truth.  All values can be overridden with environment variables — useful for
running different models on different machines without editing code.

Phase 2 note: when the task-planner spawns specialist sub-agents, each agent
can construct its own LLMConfig pointing at a different model/endpoint by
setting the relevant env vars before importing this module.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    ollama_url: str
    model: str
    num_predict: int
    stream_timeout_connect_s: float
    stream_timeout_read_s: float


LLM_CONFIG = LLMConfig(
    ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"),
    model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
    num_predict=int(os.getenv("LLM_NUM_PREDICT", "512")),
    stream_timeout_connect_s=float(os.getenv("LLM_TIMEOUT_CONNECT", "10")),
    stream_timeout_read_s=float(os.getenv("LLM_TIMEOUT_READ", "90")),
)
