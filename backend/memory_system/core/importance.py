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