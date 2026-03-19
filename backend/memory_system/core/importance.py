def calculate_importance(memory_type: str, text: str) -> float:
    score = 5.0

    if memory_type == "ActionMemory":
        score += 1.0

    if memory_type in ["ArchitectureMemory", "DecisionMemory"]:
        score += 2

    if "error" in text.lower():
        score += 1

    if "deploy" in text.lower():
        score += 1

    return min(score, 10)