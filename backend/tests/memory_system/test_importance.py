import math
from datetime import datetime, timedelta


def _iso(days_ago: int) -> str:
    return (datetime.utcnow() - timedelta(days=days_ago)).isoformat()


def test_fresh_memory_keeps_full_importance():
    from memory_system.core.importance import calculate_effective_importance
    result = calculate_effective_importance(8.0, "ActionMemory", _iso(0))
    assert result > 7.9  # almost no decay on day 0


def test_one_half_life_halves_importance():
    from memory_system.core.importance import calculate_effective_importance
    # ActionMemory half-life = 30 days
    result = calculate_effective_importance(8.0, "ActionMemory", _iso(30))
    assert 3.9 < result < 4.1  # ≈ 4.0


def test_consolidated_memory_never_decays():
    from memory_system.core.importance import calculate_effective_importance
    result = calculate_effective_importance(8.0, "ConsolidatedMemory", _iso(365))
    assert result == 8.0


def test_unknown_type_uses_default_half_life():
    from memory_system.core.importance import calculate_effective_importance
    # default = 60 days; at 60 days should be ~50% of stored
    result = calculate_effective_importance(6.0, "UnknownMemory", _iso(60))
    assert 2.9 < result < 3.1


def test_personal_memory_decays_slowly():
    from memory_system.core.importance import calculate_effective_importance
    # PersonalMemory half-life = 180 days; at 90 days ≈ 70% remaining
    result = calculate_effective_importance(8.0, "PersonalMemory", _iso(90))
    expected = 8.0 * math.pow(0.5, 90 / 180)
    assert abs(result - expected) < 0.01


def test_malformed_date_returns_stored_importance():
    from memory_system.core.importance import calculate_effective_importance
    result = calculate_effective_importance(5.0, "ActionMemory", "not-a-date")
    assert result == 5.0
