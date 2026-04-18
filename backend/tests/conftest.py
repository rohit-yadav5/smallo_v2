import sys
import os
import sqlite3
import pytest

# Make backend/ importable from tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Provide a fresh SQLite database with the full schema applied."""
    from memory_system.db import connection as conn_module
    from memory_system.db.init_db import initialize_database

    db_file = tmp_path / "test_memory.db"
    monkeypatch.setattr(conn_module, "DB_PATH", db_file)
    initialize_database(reset=False)
    return db_file


def _row_factory_conn(path: str):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@pytest.fixture
def mock_ask_llm(monkeypatch):
    """Replace ask_llm with a stub that returns a predictable string."""
    def _stub(prompt, system_suffix=""):
        return "stub response"
    import llm.main_llm as llm_mod
    monkeypatch.setattr(llm_mod, "ask_llm", _stub, raising=False)
    # Also patch the top-level llm module alias used by memory modules
    import importlib
    try:
        llm_alias = importlib.import_module("llm")
        monkeypatch.setattr(llm_alias, "ask_llm", _stub, raising=False)
    except Exception:
        pass
    return _stub
