def calculate_importance(memory_type: str, text: str) -> float:
    score = 5.0

    if memory_type == "ActionMemory":
        score += 1.0

    if memory_type in ["ArchitectureMemory", "DecisionMemory"]:
        score += 2.0

    if memory_type == "PlannerMemory":
        score += 2.0   # plan completions rank high — key for follow-up queries

    if "error" in text.lower():
        score += 1.0

    if "deploy" in text.lower():
        score += 1.0

    return min(score, 10)


import math
from datetime import datetime


def calculate_effective_importance(
    stored_importance: float,
    memory_type: str,
    created_at_iso: str,
) -> float:
    """Compute time-decayed importance for ranking.

    Stored value in SQLite is never modified — only the returned value is used
    for ranking so decay is always reversible.
    """
    from config.llm import DECAY_HALF_LIFE, DECAY_HALF_LIFE_DEFAULT

    half_life = DECAY_HALF_LIFE.get(memory_type, DECAY_HALF_LIFE_DEFAULT)
    if half_life is None:
        return stored_importance

    try:
        age_days = max(0, (datetime.utcnow() - datetime.fromisoformat(created_at_iso)).days)
    except Exception:
        return stored_importance

    decay_factor = math.pow(0.5, age_days / half_life)
    return stored_importance * decay_factor