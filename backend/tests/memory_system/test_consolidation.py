import numpy as np
from unittest.mock import patch


def test_validate_consolidation_accepts_good_summary(monkeypatch):
    monkeypatch.setattr(
        "memory_system.lifecycle.consolidator.generate_embedding_vector",
        lambda text: np.ones(384, dtype="float32")
    )
    from memory_system.lifecycle.consolidator import validate_consolidation
    assert validate_consolidation("good summary", ["source a", "source b", "source c"])


def test_validate_consolidation_rejects_bad_summary(monkeypatch):
    call_n = [0]
    def varying_vec(text):
        v = np.zeros(384, dtype="float32")
        v[call_n[0] % 384] = 1.0
        call_n[0] += 1
        return v
    monkeypatch.setattr(
        "memory_system.lifecycle.consolidator.generate_embedding_vector", varying_vec
    )
    from memory_system.lifecycle.consolidator import validate_consolidation
    result = validate_consolidation("completely unrelated text", ["source a", "source b", "source c"])
    assert result is False


def test_consolidation_threshold_comes_from_config():
    from config.llm import CONSOLIDATION_SIMILARITY_THRESHOLD
    from memory_system.lifecycle.consolidator import SIMILARITY_THRESHOLD
    assert SIMILARITY_THRESHOLD == CONSOLIDATION_SIMILARITY_THRESHOLD
