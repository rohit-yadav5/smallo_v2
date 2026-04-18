"""tests/adapters/test_browser_use_adapter.py

TDD tests for the browser-use adapter.

All tests mock the Agent class and browser_use.llm.openai.chat so no real
browser, Ollama, or torch imports are triggered.
"""

import os
import sys

# Make backend importable without installing it as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_agent(final_result_value="Task done."):
    """Return a mock Agent instance whose run() returns an AgentHistoryList-like object."""
    history = MagicMock()
    history.final_result.return_value = final_result_value
    history.number_of_steps.return_value = 3

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=history)
    return mock_agent, history


def _make_browser_use_module(agent_instance=None):
    """Return a fake browser_use module with Agent that returns agent_instance."""
    mock_module = MagicMock()
    if agent_instance is not None:
        mock_module.Agent = MagicMock(return_value=agent_instance)
    else:
        mock_agent, _ = _make_mock_agent()
        mock_module.Agent = MagicMock(return_value=mock_agent)
    return mock_module


def _make_mock_chat_openai_class():
    """Return a mock ChatOpenAI class that records constructor kwargs."""
    instances = []

    class MockChatOpenAI:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)
            instances.append(self)

    MockChatOpenAI.instances = instances
    return MockChatOpenAI


def _make_llm_chat_module(chat_openai_cls=None):
    """Return a fake browser_use.llm.openai.chat module."""
    mock_module = MagicMock()
    mock_module.ChatOpenAI = chat_openai_cls or _make_mock_chat_openai_class()
    return mock_module


def _sys_modules_patch(browser_use_module=None, llm_chat_module=None):
    """Build the sys.modules dict patch for browser_use and its LLM sub-modules.

    Args:
        browser_use_module: fake top-level browser_use module (provides Agent)
        llm_chat_module:    fake browser_use.llm.openai.chat module (provides ChatOpenAI)
                            If None, a default mock is created automatically.
    """
    patches = {}
    if browser_use_module is not None:
        patches["browser_use"] = browser_use_module

    chat_mod = llm_chat_module or _make_llm_chat_module()
    patches["browser_use.llm"] = MagicMock()
    patches["browser_use.llm.openai"] = MagicMock()
    patches["browser_use.llm.openai.chat"] = chat_mod
    return patches


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flush_adapter():
    """Purge cached import of browser_use_adapter so each test gets a fresh module."""
    for mod in list(sys.modules.keys()):
        if "browser_use_adapter" in mod:
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# Test: tool registration
# ---------------------------------------------------------------------------

def test_web_task_registered_in_registry(flush_adapter):
    """After importing the adapter, 'web_task' must appear in registry.names()."""
    mods = _sys_modules_patch(browser_use_module=_make_browser_use_module())
    with patch.dict("sys.modules", mods):
        import adapters.browser_use_adapter  # noqa: F401 — side-effect import

        from tools.registry import registry
        assert "web_task" in registry.names()


# ---------------------------------------------------------------------------
# Test: missing task returns Error:
# ---------------------------------------------------------------------------

def test_missing_task_returns_error(flush_adapter):
    """_web_task({}) must return a string starting with 'Error:'."""
    mods = _sys_modules_patch(browser_use_module=_make_browser_use_module())
    with patch.dict("sys.modules", mods):
        import adapters.browser_use_adapter as ada

        result = asyncio.run(ada._web_task({}))
        assert result.startswith("Error:"), f"Expected 'Error:...' got: {result!r}"


def test_empty_task_returns_error(flush_adapter):
    """_web_task({'task': ''}) must return a string starting with 'Error:'."""
    mods = _sys_modules_patch(browser_use_module=_make_browser_use_module())
    with patch.dict("sys.modules", mods):
        import adapters.browser_use_adapter as ada

        result = asyncio.run(ada._web_task({"task": ""}))
        assert result.startswith("Error:"), f"Expected 'Error:...' got: {result!r}"


# ---------------------------------------------------------------------------
# Test: successful run calls agent.run() and returns string result
# ---------------------------------------------------------------------------

def test_successful_task_returns_result(flush_adapter):
    """_web_task({'task': 'search for X'}) calls agent.run() and returns the result."""
    mock_agent, history = _make_mock_agent("Found information about X.")
    mods = _sys_modules_patch(browser_use_module=_make_browser_use_module(agent_instance=mock_agent))
    with patch.dict("sys.modules", mods):
        import adapters.browser_use_adapter as ada

        result = asyncio.run(ada._web_task({"task": "search for X"}))

    mock_agent.run.assert_awaited_once()
    assert isinstance(result, str)
    assert len(result) > 0

    # on_step_end must be passed as a keyword argument to agent.run()
    call_kwargs = mock_agent.run.call_args.kwargs
    assert "on_step_end" in call_kwargs
    assert callable(call_kwargs["on_step_end"])


def test_successful_task_result_contains_final_result(flush_adapter):
    """When final_result() returns a non-None string, it appears in the handler return."""
    mock_agent, history = _make_mock_agent("Specific result string.")
    mods = _sys_modules_patch(browser_use_module=_make_browser_use_module(agent_instance=mock_agent))
    with patch.dict("sys.modules", mods):
        import adapters.browser_use_adapter as ada

        result = asyncio.run(ada._web_task({"task": "do something"}))

    assert "Specific result string." in result


# ---------------------------------------------------------------------------
# Test: agent.run() raises → returns failure string
# ---------------------------------------------------------------------------

def test_agent_run_exception_returns_failure_string(flush_adapter):
    """If agent.run() raises, _web_task should return a string containing 'failed'."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=RuntimeError("browser crashed"))
    mods = _sys_modules_patch(browser_use_module=_make_browser_use_module(agent_instance=mock_agent))
    with patch.dict("sys.modules", mods):
        import adapters.browser_use_adapter as ada

        result = asyncio.run(ada._web_task({"task": "do something"}))

    assert "failed" in result.lower(), f"Expected failure string, got: {result!r}"


# ---------------------------------------------------------------------------
# Test: _make_llm() returns ChatOpenAI pointed at localhost:11434
# ---------------------------------------------------------------------------

def test_make_llm_uses_local_ollama(flush_adapter):
    """_make_llm() must create a ChatOpenAI with base_url containing 'localhost:11434'."""
    MockChatOpenAI = _make_mock_chat_openai_class()
    llm_chat_mod = _make_llm_chat_module(chat_openai_cls=MockChatOpenAI)

    mods = _sys_modules_patch(
        browser_use_module=_make_browser_use_module(),
        llm_chat_module=llm_chat_mod,
    )
    with patch.dict("sys.modules", mods):
        import adapters.browser_use_adapter as ada

        llm = ada._make_llm()

    # The mock records what was passed to ChatOpenAI(...)
    assert "localhost:11434" in str(getattr(llm, "base_url", "")), (
        f"Expected localhost:11434 in base_url, got: {getattr(llm, 'base_url', None)!r}"
    )


# ---------------------------------------------------------------------------
# Test: final_result() returning None falls back gracefully
# ---------------------------------------------------------------------------

def test_none_final_result_returns_nonempty_string(flush_adapter):
    """When final_result() returns None, _web_task still returns a non-empty string."""
    mock_agent, history = _make_mock_agent(final_result_value=None)
    mods = _sys_modules_patch(browser_use_module=_make_browser_use_module(agent_instance=mock_agent))
    with patch.dict("sys.modules", mods):
        import adapters.browser_use_adapter as ada

        result = asyncio.run(ada._web_task({"task": "something"}))

    assert isinstance(result, str)
    assert len(result) > 0
