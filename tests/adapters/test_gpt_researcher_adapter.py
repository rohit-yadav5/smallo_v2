"""tests/adapters/test_gpt_researcher_adapter.py

TDD tests for the GPT-Researcher adapter.

All tests mock GPTResearcher so no network or Ollama calls are made.
"""

import os
import sys

# Make backend importable without installing it as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flush_adapter():
    """Purge any cached import of gpt_researcher_adapter so each test gets a
    fresh module execution (re-runs env-var setdefault + registry.register).
    """
    for mod in list(sys.modules.keys()):
        if "gpt_researcher_adapter" in mod:
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# Test: tool registration
# ---------------------------------------------------------------------------

def test_deep_research_registered_in_registry(flush_adapter):
    """After importing the adapter, 'deep_research' must appear in registry.names()."""
    with patch.dict("sys.modules", {"gpt_researcher": MagicMock()}):
        import adapters.gpt_researcher_adapter  # noqa: F401 — side-effect import
        importlib.reload(adapters.gpt_researcher_adapter)

        from tools.registry import registry
        assert "deep_research" in registry.names()


# ---------------------------------------------------------------------------
# Test: missing topic returns Error:
# ---------------------------------------------------------------------------

def test_missing_topic_returns_error(flush_adapter):
    """_deep_research({}) must return a string starting with 'Error:'."""
    with patch.dict("sys.modules", {"gpt_researcher": MagicMock()}):
        import adapters.gpt_researcher_adapter as ada
        importlib.reload(ada)

        result = asyncio.run(ada._deep_research({}))
        assert result.startswith("Error:"), f"Expected 'Error:...' got: {result!r}"


# ---------------------------------------------------------------------------
# Test: successful research calls conduct_research + write_report
# ---------------------------------------------------------------------------

def test_successful_research_returns_report(flush_adapter):
    """_deep_research({'topic': 'X'}) should call conduct_research and write_report
    and return the report string."""
    mock_researcher_instance = MagicMock()
    mock_researcher_instance.conduct_research = AsyncMock(return_value=None)
    mock_researcher_instance.write_report = AsyncMock(return_value="# Report on X\n\nDetails here.")

    mock_gpt_researcher_module = MagicMock()
    mock_gpt_researcher_module.GPTResearcher = MagicMock(return_value=mock_researcher_instance)

    with patch.dict("sys.modules", {"gpt_researcher": mock_gpt_researcher_module}):
        import adapters.gpt_researcher_adapter as ada
        importlib.reload(ada)

        result = asyncio.run(ada._deep_research({"topic": "X"}))

    mock_researcher_instance.conduct_research.assert_awaited_once()
    mock_researcher_instance.write_report.assert_awaited_once()
    assert "Report on X" in result or len(result) > 0


# ---------------------------------------------------------------------------
# Test: conduct_research raises → returns failure string
# ---------------------------------------------------------------------------

def test_conduct_research_exception_returns_failure_string(flush_adapter):
    """If conduct_research raises, _deep_research should return a string containing
    'failed' or 'Research failed' (case-insensitive)."""
    mock_researcher_instance = MagicMock()
    mock_researcher_instance.conduct_research = AsyncMock(
        side_effect=RuntimeError("connection refused")
    )
    mock_researcher_instance.write_report = AsyncMock(return_value="should not reach here")

    mock_gpt_researcher_module = MagicMock()
    mock_gpt_researcher_module.GPTResearcher = MagicMock(return_value=mock_researcher_instance)

    with patch.dict("sys.modules", {"gpt_researcher": mock_gpt_researcher_module}):
        import adapters.gpt_researcher_adapter as ada
        importlib.reload(ada)

        result = asyncio.run(ada._deep_research({"topic": "AI"}))

    assert "failed" in result.lower() or "research failed" in result.lower(), (
        f"Expected failure string, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Test: 'query' alias for 'topic'
# ---------------------------------------------------------------------------

def test_query_alias_accepted(flush_adapter):
    """_deep_research({'query': 'X'}) should work the same as using 'topic'."""
    mock_researcher_instance = MagicMock()
    mock_researcher_instance.conduct_research = AsyncMock(return_value=None)
    mock_researcher_instance.write_report = AsyncMock(return_value="# Alias report")

    mock_gpt_researcher_module = MagicMock()
    mock_gpt_researcher_module.GPTResearcher = MagicMock(return_value=mock_researcher_instance)

    with patch.dict("sys.modules", {"gpt_researcher": mock_gpt_researcher_module}):
        import adapters.gpt_researcher_adapter as ada
        importlib.reload(ada)

        result = asyncio.run(ada._deep_research({"query": "X"}))

    mock_researcher_instance.conduct_research.assert_awaited_once()
    assert "Alias report" in result or len(result) > 0
