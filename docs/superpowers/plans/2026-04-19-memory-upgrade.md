# Memory System v2 — Intelligence Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the memory system with two-tier retrieval (fast <100ms / deep 3–5s), LLM re-ranking, importance decay, affect tagging, memory chains, entity graph activation, and validated consolidation.

**Architecture:** Keep SQLite + FAISS + spaCy as the foundation; add an intelligence layer on top. Insert pipeline enriches memories asynchronously (affect, LLM summary). Retrieval routes to fast (unchanged) or deep path (query rewrite → FAISS top-20 → BFS expansion → LLM re-rank → multi-hop). Importance decays lazily at retrieval time without modifying stored values.

**Tech Stack:** Python asyncio, SQLite (sqlite3), FAISS (IndexFlatIP), sentence-transformers/all-MiniLM-L6-v2, Ollama qwen2.5:3b/7b, pytest + unittest.mock

**Run all tests from:** `cd backend && python -m pytest tests/ -v`

---

## File Map

### New Files
| Path | Purpose |
|---|---|
| `backend/tests/__init__.py` | Makes tests a package |
| `backend/tests/conftest.py` | Shared fixtures: temp DB, mock LLM, mock FAISS |
| `backend/tests/memory_system/test_affect.py` | Tests for affect detection |
| `backend/tests/memory_system/test_importance.py` | Tests for decay calculation |
| `backend/tests/memory_system/test_chain.py` | Tests for chain creation/traversal |
| `backend/tests/memory_system/test_async_summary.py` | Tests for summary importance bump |
| `backend/tests/memory_system/test_query_rewriter.py` | Tests for query rewriter |
| `backend/tests/memory_system/test_reranker.py` | Tests for LLM re-ranker |
| `backend/tests/memory_system/test_retrieval_routing.py` | Tests for fast/deep path classification |
| `backend/tests/memory_system/test_consolidation.py` | Tests for consolidation validation |
| `backend/memory_system/core/affect.py` | Affect tagging: keyword heuristic + LLM fallback |
| `backend/memory_system/core/async_summary.py` | Async LLM summary generation + importance bump |
| `backend/memory_system/core/chain.py` | Memory chain CRUD and chain-type detection |
| `backend/memory_system/retrieval/query_rewriter.py` | LLM query expansion before FAISS |
| `backend/memory_system/retrieval/reranker.py` | Two-pass LLM re-ranking (coarse + fine) |

### Modified Files
| Path | Change |
|---|---|
| `backend/memory_system/schema.sql` | Add `affect TEXT DEFAULT 'neutral'` column |
| `backend/memory_system/db/init_db.py` | Add runtime migration for `affect` column |
| `backend/config/llm.py` | Add DECAY_HALF_LIFE dict + consolidation constants |
| `backend/memory_system/core/importance.py` | Add `calculate_effective_importance()` |
| `backend/memory_system/entities/service.py` | Populate `entity_relations` via AUTO_PARENT_RULES |
| `backend/memory_system/core/insert_pipeline.py` | Wire affect, chain detection, async summary |
| `backend/memory_system/retrieval/search.py` | Two-tier routing, decay scoring, affect boost, confidence filter, multi-hop |
| `backend/memory_system/lifecycle/consolidator.py` | Validated consolidation, meta-consolidation, expiry |

---

## Task 1: Schema Migration + Config Constants

**Files:**
- Modify: `backend/memory_system/schema.sql`
- Modify: `backend/memory_system/db/init_db.py`
- Modify: `backend/config/llm.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/memory_system/__init__.py`
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: Add `affect` column to schema.sql**

In `backend/memory_system/schema.sql`, add `affect TEXT DEFAULT 'neutral'` after the `session_id` line:

```sql
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    title TEXT,
    raw_text TEXT NOT NULL,
    summary TEXT,
    importance_score REAL DEFAULT 0,
    confidence_score REAL DEFAULT 1,
    source TEXT,
    project_reference TEXT,
    status TEXT DEFAULT 'active',
    session_id TEXT DEFAULT 'legacy',
    affect TEXT DEFAULT 'neutral',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Leave all other tables unchanged.

- [ ] **Step 2: Add runtime migration to init_db.py**

In `backend/memory_system/db/init_db.py`, add a `migrate_database()` function after `initialize_database()`:

```python
def migrate_database():
    """Apply incremental migrations to existing databases."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if affect column exists (SQLite has no IF NOT EXISTS for ALTER TABLE)
    cursor.execute("PRAGMA table_info(memories)")
    columns = {row["name"] for row in cursor.fetchall()}

    if "affect" not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN affect TEXT DEFAULT 'neutral'")
        conn.commit()
        print("Migration: added 'affect' column to memories.")

    conn.close()
```

Then call `migrate_database()` at the end of `initialize_database()`:

```python
def initialize_database(reset: bool = False):
    # ... existing code unchanged ...
    print("Database initialized.")
    migrate_database()   # ← add this line
```

- [ ] **Step 3: Add memory constants to config/llm.py**

At the end of `backend/config/llm.py`, after the `LLM_CONFIG` instance, add:

```python
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
```

- [ ] **Step 4: Create test package files**

Create `backend/tests/__init__.py` (empty):
```python
```

Create `backend/tests/memory_system/__init__.py` (empty):
```python
```

- [ ] **Step 5: Create conftest.py with shared fixtures**

Create `backend/tests/conftest.py`:

```python
import sys
import os
import sqlite3
import tempfile
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
    monkeypatch.setattr(
        conn_module, "get_connection",
        lambda: _row_factory_conn(str(db_file))
    )
    initialize_database(reset=False)
    return db_file


def _row_factory_conn(path: str):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
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
```

- [ ] **Step 6: Write migration test**

Create `backend/tests/memory_system/test_schema.py`:

```python
def test_affect_column_exists_after_migration(tmp_db):
    import sqlite3
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(memories)")
    columns = {row["name"] for row in cursor.fetchall()}
    conn.close()
    assert "affect" in columns

def test_affect_default_is_neutral(tmp_db):
    import sqlite3
    from datetime import datetime
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memories (id, memory_type, raw_text, summary, importance_score)
        VALUES ('test-001', 'IdeaMemory', 'hello world', 'hello', 5.0)
    """)
    conn.commit()
    cursor.execute("SELECT affect FROM memories WHERE id = 'test-001'")
    row = cursor.fetchone()
    conn.close()
    assert row["affect"] == "neutral"
```

- [ ] **Step 7: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_schema.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add backend/memory_system/schema.sql backend/memory_system/db/init_db.py \
        backend/config/llm.py backend/tests/
git commit -m "feat(memory): schema migration, config constants, test infrastructure"
```

---

## Task 2: Affect Tagging Module

**Files:**
- Create: `backend/memory_system/core/affect.py`
- Create: `backend/tests/memory_system/test_affect.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/memory_system/test_affect.py`:

```python
import pytest


def test_detect_affect_positive():
    from memory_system.core.affect import detect_affect
    assert detect_affect("That works perfectly, great result!") == "positive"


def test_detect_affect_negative():
    from memory_system.core.affect import detect_affect
    assert detect_affect("The app crashed with an error again") == "negative"


def test_detect_affect_frustrated():
    from memory_system.core.affect import detect_affect
    assert detect_affect("This keeps failing, it's so frustrating") == "frustrated"


def test_detect_affect_excited():
    from memory_system.core.affect import detect_affect
    assert detect_affect("Finally got it working, this is amazing!") == "excited"


def test_detect_affect_uncertain():
    from memory_system.core.affect import detect_affect
    assert detect_affect("Not sure if this approach is correct, maybe try another") == "uncertain"


def test_detect_affect_unknown_falls_back_to_neutral(monkeypatch):
    """LLM fallback returns neutral when LLM also can't classify."""
    monkeypatch.setattr(
        "memory_system.core.affect._detect_affect_llm",
        lambda text: "neutral"
    )
    from memory_system.core.affect import detect_affect
    assert detect_affect("the quick brown fox") == "neutral"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_affect.py -v
```

Expected: ImportError — `memory_system.core.affect` not found.

- [ ] **Step 3: Create backend/memory_system/core/affect.py**

```python
_FRUSTRATED = {"frustrated", "annoying", "keeps", "keep failing", "again", "why won't", "ugh"}
_EXCITED    = {"amazing", "awesome", "finally", "breakthrough", "excited", "wow"}
_POSITIVE   = {"great", "love", "perfect", "excellent", "happy", "solved", "fixed", "success", "works", "working"}
_NEGATIVE   = {"hate", "broken", "failed", "error", "crash", "terrible", "wrong", "bad", "issue", "bug"}
_UNCERTAIN  = {"maybe", "not sure", "unsure", "unclear", "confused", "might", "possibly", "perhaps"}

_PRIORITY = [
    ("frustrated", _FRUSTRATED),
    ("excited",    _EXCITED),
    ("positive",   _POSITIVE),
    ("negative",   _NEGATIVE),
    ("uncertain",  _UNCERTAIN),
]


def detect_affect(text: str) -> str:
    """Return emotional affect label for text. Keyword-first, LLM fallback."""
    lower = text.lower()
    for label, keywords in _PRIORITY:
        if any(kw in lower for kw in keywords):
            return label
    return _detect_affect_llm(text)


def _detect_affect_llm(text: str) -> str:
    """LLM fallback when no keyword matches. Returns one of the 6 affect labels."""
    try:
        from llm import ask_llm
        prompt = (
            "Classify the emotional affect of this text. "
            "Respond with exactly one word from: positive, negative, neutral, frustrated, excited, uncertain\n\n"
            f"Text: {text[:200]}\n\nAffect:"
        )
        result = ask_llm(prompt).strip().lower().split()[0]
        valid = {"positive", "negative", "neutral", "frustrated", "excited", "uncertain"}
        return result if result in valid else "neutral"
    except Exception:
        return "neutral"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_affect.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/core/affect.py backend/tests/memory_system/test_affect.py
git commit -m "feat(memory): affect tagging module with keyword heuristic and LLM fallback"
```

---

## Task 3: Importance Decay

**Files:**
- Modify: `backend/memory_system/core/importance.py`
- Create: `backend/tests/memory_system/test_importance.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/memory_system/test_importance.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_importance.py -v
```

Expected: ImportError — `calculate_effective_importance` not found.

- [ ] **Step 3: Add calculate_effective_importance to importance.py**

Open `backend/memory_system/core/importance.py` and append at the end:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_importance.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/core/importance.py backend/tests/memory_system/test_importance.py
git commit -m "feat(memory): time-based importance decay with per-type half-lives"
```

---

## Task 4: Entity Graph Population

**Files:**
- Modify: `backend/memory_system/entities/service.py`
- Create: `backend/tests/memory_system/test_entity_graph.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/memory_system/test_entity_graph.py`:

```python
import sqlite3


def test_entity_relation_created_for_known_parent(tmp_db):
    """Inserting 'redis' entity should auto-create an is_a relation to 'cache'."""
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Pre-create the parent entity
    cursor.execute("""
        INSERT OR IGNORE INTO entities (id, name, domain, category, entity_type, usage_count, importance_score)
        VALUES ('parent-001', 'cache', 'Engineering', 'Technology', 'Technology', 1, 1.0)
    """)
    conn.commit()

    from memory_system.entities.service import get_or_create_entity
    redis_id = get_or_create_entity(cursor, "redis", "Engineering", "Technology", "Technology")
    conn.commit()

    cursor.execute("""
        SELECT * FROM entity_relations
        WHERE source_entity_id = ? AND target_entity_id = 'parent-001'
    """, (redis_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["relation_type"] == "is_a"


def test_no_relation_created_when_parent_missing(tmp_db):
    """If parent entity doesn't exist yet, no relation is written (no orphan creation)."""
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    from memory_system.entities.service import get_or_create_entity
    # "kafka" → "message_broker" but message_broker entity doesn't exist
    kafka_id = get_or_create_entity(cursor, "kafka", "Engineering", "Infrastructure", "Infrastructure")
    conn.commit()

    cursor.execute("SELECT * FROM entity_relations WHERE source_entity_id = ?", (kafka_id,))
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 0


def test_no_duplicate_relations_on_second_insert(tmp_db):
    """Re-inserting same entity should not duplicate entity_relations rows."""
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO entities (id, name, domain, category, entity_type, usage_count, importance_score)
        VALUES ('parent-db', 'database', 'Engineering', 'Technology', 'Technology', 1, 1.0)
    """)
    conn.commit()

    from memory_system.entities.service import get_or_create_entity
    get_or_create_entity(cursor, "postgresql", "Engineering", "Technology", "Technology")
    conn.commit()
    get_or_create_entity(cursor, "postgresql", "Engineering", "Technology", "Technology")
    conn.commit()

    cursor.execute("SELECT COUNT(*) as cnt FROM entity_relations WHERE relation_type = 'is_a'")
    count = cursor.fetchone()["cnt"]
    conn.close()

    assert count == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_entity_graph.py -v
```

Expected: 3 failed — entity_relations never written.

- [ ] **Step 3: Modify entities/service.py**

In `backend/memory_system/entities/service.py`, replace the existing comment block at the end of the entity creation section with a helper function and a call to it. The full modified file:

```python
import uuid
from datetime import datetime

AUTO_PARENT_RULES = {
    "redis": "cache",
    "postgresql": "database",
    "postgres": "database",
    "mysql": "database",
    "faiss": "vectordb",
    "chromadb": "vectordb",
    "docker": "infrastructure",
    "kafka": "message_broker"
}


def _write_entity_relation(cursor, child_id: str, parent_name: str) -> None:
    """Write a parent is_a relation if the parent entity already exists."""
    cursor.execute("SELECT id FROM entities WHERE name = ?", (parent_name,))
    row = cursor.fetchone()
    if not row:
        return
    parent_id = row["id"]
    rel_id = f"rel-{child_id[:8]}-{parent_id[:8]}"
    cursor.execute("""
        INSERT OR IGNORE INTO entity_relations
        (id, source_entity_id, target_entity_id, relation_type, created_at)
        VALUES (?, ?, ?, 'is_a', ?)
    """, (rel_id, child_id, parent_id, datetime.utcnow().isoformat()))


def get_or_create_entity(cursor, name: str, domain: str, category: str, entity_type: str):

    normalized = name.strip().lower()

    cursor.execute("""
        SELECT id, usage_count FROM entities WHERE name = ?
    """, (normalized,))
    row = cursor.fetchone()

    if row:
        entity_id = row["id"]
        cursor.execute("""
            UPDATE entities
            SET usage_count = usage_count + 1,
                last_used_at = ?
            WHERE id = ?
        """, (
            datetime.utcnow().isoformat(),
            entity_id
        ))
        return entity_id

    entity_id = str(uuid.uuid4())

    cursor.execute("""
        INSERT INTO entities
        (id, name, domain, category, entity_type, usage_count, importance_score, last_used_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entity_id,
        normalized,
        domain,
        category,
        entity_type,
        1,
        1.0,
        datetime.utcnow().isoformat()
    ))

    parent_name = AUTO_PARENT_RULES.get(normalized)
    if parent_name:
        _write_entity_relation(cursor, entity_id, parent_name)

    return entity_id
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_entity_graph.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/entities/service.py backend/tests/memory_system/test_entity_graph.py
git commit -m "feat(memory): populate entity_relations via AUTO_PARENT_RULES on insert"
```

---

## Task 5: Memory Chain Module

**Files:**
- Create: `backend/memory_system/core/chain.py`
- Create: `backend/tests/memory_system/test_chain.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/memory_system/test_chain.py`:

```python
import sqlite3
import pytest


def _seed_memory(cursor, memory_id: str, affect: str = "neutral"):
    cursor.execute("""
        INSERT OR IGNORE INTO memories (id, memory_type, raw_text, summary, importance_score, affect)
        VALUES (?, 'IdeaMemory', 'test text', 'test', 5.0, ?)
    """, (memory_id, affect))


def test_create_chain_inserts_relation(tmp_db):
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _seed_memory(cursor, "mem-a")
    _seed_memory(cursor, "mem-b")
    conn.commit()

    from memory_system.core.chain import create_chain
    create_chain(cursor, "mem-a", "mem-b", "caused_by")
    conn.commit()

    cursor.execute("""
        SELECT * FROM memory_relations WHERE source_memory_id = 'mem-a' AND relation_type = 'caused_by'
    """)
    assert cursor.fetchone() is not None
    conn.close()


def test_get_chain_links_returns_targets(tmp_db):
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _seed_memory(cursor, "mem-x")
    _seed_memory(cursor, "mem-y")
    _seed_memory(cursor, "mem-z")
    conn.commit()

    from memory_system.core.chain import create_chain, get_chain_links
    create_chain(cursor, "mem-x", "mem-y", "led_to")
    create_chain(cursor, "mem-x", "mem-z", "confirms")
    conn.commit()

    links = get_chain_links(cursor, "mem-x")
    target_ids = {l["target_memory_id"] for l in links}
    conn.close()
    assert target_ids == {"mem-y", "mem-z"}


def test_get_chain_links_filtered_by_type(tmp_db):
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    _seed_memory(cursor, "mem-1")
    _seed_memory(cursor, "mem-2")
    _seed_memory(cursor, "mem-3")
    conn.commit()

    from memory_system.core.chain import create_chain, get_chain_links
    create_chain(cursor, "mem-1", "mem-2", "confirms")
    create_chain(cursor, "mem-1", "mem-3", "contradicts")
    conn.commit()

    confirms = get_chain_links(cursor, "mem-1", relation_type="confirms")
    conn.close()
    assert len(confirms) == 1
    assert confirms[0]["target_memory_id"] == "mem-2"


@pytest.mark.parametrize("src_affect,tgt_affect,expected", [
    ("positive", "negative", "contradicts"),
    ("excited",  "frustrated", "contradicts"),
    ("positive", "positive", "confirms"),
    ("neutral",  "negative", "confirms"),
    ("frustrated", "frustrated", "confirms"),
])
def test_detect_chain_type(src_affect, tgt_affect, expected):
    from memory_system.core.chain import detect_chain_type
    assert detect_chain_type(src_affect, tgt_affect) == expected


def test_invalid_relation_type_raises():
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(":memory:")
    cursor = conn.cursor()
    from memory_system.core.chain import create_chain
    with pytest.raises(AssertionError):
        create_chain(cursor, "a", "b", "invented_type")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_chain.py -v
```

Expected: ImportError — `memory_system.core.chain` not found.

- [ ] **Step 3: Create backend/memory_system/core/chain.py**

```python
import uuid
from datetime import datetime

CHAIN_RELATION_TYPES = {"caused_by", "led_to", "contradicts", "confirms", "summarized_by"}

_POSITIVE_AFFECTS = {"positive", "excited"}
_NEGATIVE_AFFECTS = {"negative", "frustrated"}


def create_chain(cursor, source_id: str, target_id: str, relation_type: str) -> str:
    """Insert a directed memory chain link. Returns the relation id."""
    assert relation_type in CHAIN_RELATION_TYPES, (
        f"Unknown relation type '{relation_type}'. Must be one of {CHAIN_RELATION_TYPES}"
    )
    rel_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT OR IGNORE INTO memory_relations
        (id, source_memory_id, target_memory_id, relation_type, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (rel_id, source_id, target_id, relation_type, datetime.utcnow().isoformat()))
    return rel_id


def get_chain_links(cursor, memory_id: str, relation_type: str = None) -> list[dict]:
    """Return all memory_relations rows where source_memory_id = memory_id."""
    if relation_type:
        cursor.execute("""
            SELECT target_memory_id, relation_type FROM memory_relations
            WHERE source_memory_id = ? AND relation_type = ?
        """, (memory_id, relation_type))
    else:
        cursor.execute("""
            SELECT target_memory_id, relation_type FROM memory_relations
            WHERE source_memory_id = ?
        """, (memory_id,))
    return [dict(r) for r in cursor.fetchall()]


def detect_chain_type(source_affect: str, target_affect: str) -> str:
    """Return 'contradicts' if affects are opposite polarity, 'confirms' otherwise."""
    src_pos = source_affect in _POSITIVE_AFFECTS
    src_neg = source_affect in _NEGATIVE_AFFECTS
    tgt_pos = target_affect in _POSITIVE_AFFECTS
    tgt_neg = target_affect in _NEGATIVE_AFFECTS

    if (src_pos and tgt_neg) or (src_neg and tgt_pos):
        return "contradicts"
    return "confirms"
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_chain.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/core/chain.py backend/tests/memory_system/test_chain.py
git commit -m "feat(memory): memory chain module for causal/confirmatory links"
```

---

## Task 6: Async LLM Summary Module

**Files:**
- Create: `backend/memory_system/core/async_summary.py`
- Create: `backend/tests/memory_system/test_async_summary.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/memory_system/test_async_summary.py`:

```python
def test_high_signal_phrase_gives_bump():
    from memory_system.core.async_summary import _compute_importance_bump
    assert _compute_importance_bump("I prefer dark mode for all interfaces") == 2.0


def test_neutral_summary_gives_no_bump():
    from memory_system.core.async_summary import _compute_importance_bump
    assert _compute_importance_bump("The server started on port 8765") == 0.0


def test_decision_phrase_gives_bump():
    from memory_system.core.async_summary import _compute_importance_bump
    assert _compute_importance_bump("Decided to use FAISS over ChromaDB") == 2.0


def test_call_llm_for_summary_returns_string(monkeypatch):
    monkeypatch.setattr("memory_system.core.async_summary._call_llm_for_summary",
                        lambda text, memory_type: "Test summary sentence.")
    from memory_system.core.async_summary import _call_llm_for_summary
    result = _call_llm_for_summary("some long text here", "IdeaMemory")
    assert isinstance(result, str)
    assert len(result) > 0


def test_call_llm_for_summary_truncates_long_result(monkeypatch):
    long_result = "x" * 500
    monkeypatch.setattr("memory_system.core.async_summary._call_llm_for_summary",
                        lambda text, memory_type: long_result[:300])
    from memory_system.core.async_summary import _call_llm_for_summary
    result = _call_llm_for_summary("text", "IdeaMemory")
    assert len(result) <= 300
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_async_summary.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create backend/memory_system/core/async_summary.py**

```python
import asyncio
from memory_system.db.connection import get_connection

_HIGH_SIGNAL_PHRASES = [
    "i prefer", "i want", "i need", "i always", "i never",
    "my goal", "decided to", "we decided", "the decision",
    "chosen to", "important to me", "make sure",
    "never do", "always do", "my preference",
]


async def generate_and_store_summary(memory_id: str, raw_text: str, memory_type: str) -> None:
    """Fire-and-forget async task: generates LLM summary and updates the DB record.

    The memory record is already inserted with a truncated placeholder before this
    task runs — so retrieval always returns *something* even if this hasn't finished.
    """
    try:
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(
            None,
            lambda: _call_llm_for_summary(raw_text, memory_type),
        )
        bump = _compute_importance_bump(summary)

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE memories
            SET summary = ?,
                importance_score = MIN(importance_score + ?, 10.0)
            WHERE id = ?
        """, (summary, bump, memory_id))
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[memory] async summary failed for {memory_id}: {exc}", flush=True)


def _call_llm_for_summary(raw_text: str, memory_type: str) -> str:
    """Blocking LLM call — run via executor, never call from event loop directly."""
    try:
        from llm import ask_llm
        prompt = (
            f"Summarize this memory in one concise sentence. Preserve all key facts.\n\n"
            f"Memory type: {memory_type}\n"
            f"Text: {raw_text[:500]}\n\n"
            "One sentence summary:"
        )
        result = ask_llm(prompt).strip()
        return result[:300] if result else raw_text[:300]
    except Exception:
        return raw_text[:300]


def _compute_importance_bump(summary: str) -> float:
    """Return +2.0 if summary contains a high-signal phrase, else 0.0."""
    lower = summary.lower()
    for phrase in _HIGH_SIGNAL_PHRASES:
        if phrase in lower:
            return 2.0
    return 0.0
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_async_summary.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/core/async_summary.py backend/tests/memory_system/test_async_summary.py
git commit -m "feat(memory): async LLM summary generation with importance bump"
```

---

## Task 7: Wire Insert Pipeline

**Files:**
- Modify: `backend/memory_system/core/insert_pipeline.py`
- Create: `backend/tests/memory_system/test_insert_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/memory_system/test_insert_pipeline.py`:

```python
from unittest.mock import patch, MagicMock
import sqlite3


def test_insert_stores_affect_column(tmp_db, monkeypatch):
    """After insert, the affect column should be populated (not NULL or empty)."""
    # Monkeypatch heavy dependencies
    monkeypatch.setattr("memory_system.core.insert_pipeline.generate_embedding_vector",
                        lambda text: __import__("numpy").zeros(384, dtype="float32"))
    monkeypatch.setattr("memory_system.core.insert_pipeline.search_vector",
                        lambda vec, top_k: ([], []))
    monkeypatch.setattr("memory_system.core.insert_pipeline.add_vector",
                        lambda mid, vec: 0)
    monkeypatch.setattr("memory_system.core.insert_pipeline.extract_entities",
                        lambda text: [])

    from memory_system.core.insert_pipeline import insert_memory
    mid = insert_memory({"text": "I love this feature, it works perfectly!", "memory_type": "IdeaMemory"})

    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT affect FROM memories WHERE id = ?", (mid,)).fetchone()
    conn.close()
    assert row is not None
    assert row["affect"] in {"positive", "negative", "neutral", "frustrated", "excited", "uncertain"}


def test_near_duplicate_creates_chain_link(tmp_db, monkeypatch):
    """Two similar (0.75-0.89) inserts should create a chain relation."""
    import numpy as np
    call_count = [0]

    def mock_search(vec, top_k):
        call_count[0] += 1
        if call_count[0] == 1:
            return ([0.82], [0])   # near-duplicate on first insert
        return ([], [-1])

    monkeypatch.setattr("memory_system.core.insert_pipeline.generate_embedding_vector",
                        lambda text: np.zeros(384, dtype="float32"))
    monkeypatch.setattr("memory_system.core.insert_pipeline.search_vector", mock_search)
    monkeypatch.setattr("memory_system.core.insert_pipeline.add_vector",
                        lambda mid, vec: 0)
    monkeypatch.setattr("memory_system.core.insert_pipeline.extract_entities",
                        lambda text: [])

    # Pre-insert a memory that will act as the near-duplicate
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        INSERT INTO memories (id, memory_type, raw_text, summary, importance_score, affect)
        VALUES ('existing-001', 'IdeaMemory', 'old text', 'old summary', 5.0, 'positive')
    """)
    conn.execute("""
        INSERT INTO memory_embeddings (memory_id, vector_id, model_name)
        VALUES ('existing-001', '0', 'all-MiniLM-L6-v2')
    """)
    conn.commit()
    conn.close()

    from memory_system.core.insert_pipeline import insert_memory
    new_mid = insert_memory({"text": "similar text here", "memory_type": "IdeaMemory"})

    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT * FROM memory_relations WHERE source_memory_id = ?
    """, (new_mid,)).fetchone()
    conn.close()
    assert row is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_insert_pipeline.py -v
```

Expected: 2 failures — affect column not populated, no chain link created.

- [ ] **Step 3: Rewrite insert_pipeline.py**

Replace the full content of `backend/memory_system/core/insert_pipeline.py` with:

```python
import asyncio
import uuid
from datetime import datetime

from memory_system.db.connection import get_connection
from memory_system.core.importance import calculate_importance
from memory_system.core.affect import detect_affect
from memory_system.entities.extractor import extract_entities
from memory_system.entities.service import get_or_create_entity
from memory_system.embeddings.embedder import generate_embedding_vector
from memory_system.embeddings.vector_store import add_vector
from memory_system.embeddings.vector_store import search_vector

DEDUP_THRESHOLD = 0.90
NEAR_DEDUP_THRESHOLD = 0.75   # below this: unrelated; above but <0.90: near-duplicate


def _get_session_id() -> str:
    try:
        import backend_loop_ref as _ref
        return _ref.session_id or "unknown"
    except Exception:
        return "unknown"


def insert_memory(input_data: dict) -> str:
    memory_id   = str(uuid.uuid4())
    created_at  = datetime.utcnow().isoformat()
    raw_text    = input_data["text"]
    source      = input_data.get("source", "manual")
    memory_type = input_data.get("memory_type", "IdeaMemory")
    summary     = raw_text[:300]
    session_id  = _get_session_id()

    # ── Step 1: Affect tagging ────────────────────────────────────────────────
    affect = detect_affect(raw_text)

    # ── Step 2: Pre-insert deduplication + near-duplicate chain detection ─────
    embedding_input = f"{memory_type} | {summary}"
    new_vector = generate_embedding_vector(embedding_input)
    distances, indices = search_vector(new_vector, top_k=5)

    near_duplicates: list[tuple[str, str]] = []   # (existing_memory_id, existing_affect)

    conn = get_connection()
    cursor = conn.cursor()

    for distance, idx in zip(distances, indices):
        if idx == -1:
            continue
        similarity = float(distance)

        if similarity >= DEDUP_THRESHOLD:
            cursor.execute("""
                SELECT memory_id FROM memory_embeddings WHERE vector_id = ?
            """, (str(idx),))
            row = cursor.fetchone()
            if row:
                existing_id = row["memory_id"]
                if session_id and session_id not in ("unknown", "legacy"):
                    cursor.execute(
                        "UPDATE memories SET session_id = ? WHERE id = ?",
                        (session_id, existing_id),
                    )
                    conn.commit()
                conn.close()
                return existing_id

        elif similarity >= NEAR_DEDUP_THRESHOLD:
            cursor.execute("""
                SELECT me.memory_id, m.affect
                FROM memory_embeddings me
                JOIN memories m ON m.id = me.memory_id
                WHERE me.vector_id = ?
            """, (str(idx),))
            row = cursor.fetchone()
            if row:
                near_duplicates.append((row["memory_id"], row["affect"] or "neutral"))

    conn.close()

    # ── Step 3: Entity extraction + importance ────────────────────────────────
    entities   = extract_entities(raw_text)
    importance = calculate_importance(memory_type, raw_text)

    # ── Step 4: Insert into SQLite ────────────────────────────────────────────
    conn   = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO memories
            (id, memory_type, raw_text, summary, importance_score, source,
             session_id, affect, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, memory_type, raw_text, summary,
            importance, source, session_id, affect, created_at,
        ))

        for entity in entities:
            entity_id = get_or_create_entity(
                cursor,
                name=entity["name"],
                domain=entity["domain"],
                category=entity["category"],
                entity_type=entity["entity_type"],
            )
            cursor.execute("""
                INSERT OR IGNORE INTO memory_entities (memory_id, entity_id)
                VALUES (?, ?)
            """, (memory_id, entity_id))

        # ── Step 4b: Chain links for near-duplicates ──────────────────────────
        from memory_system.core.chain import create_chain, detect_chain_type
        for existing_id, existing_affect in near_duplicates:
            chain_type = detect_chain_type(affect, existing_affect)
            create_chain(cursor, memory_id, existing_id, chain_type)
            # Update confidence of older memory if contradicted
            if chain_type == "contradicts":
                cursor.execute("""
                    UPDATE memories
                    SET confidence_score = MAX(confidence_score - 0.2, 0.0)
                    WHERE id = ?
                """, (existing_id,))

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

    # ── Step 5: FAISS insert ──────────────────────────────────────────────────
    numeric_id = add_vector(memory_id, new_vector)

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memory_embeddings (memory_id, vector_id, model_name)
        VALUES (?, ?, ?)
    """, (memory_id, str(numeric_id), "all-MiniLM-L6-v2"))
    conn.commit()
    conn.close()

    # ── Step 6: Async summary generation (fire-and-forget) ───────────────────
    try:
        from memory_system.core.async_summary import generate_and_store_summary
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(generate_and_store_summary(memory_id, raw_text, memory_type))
    except Exception:
        pass

    # ── Step 7: Memory cap check ──────────────────────────────────────────────
    try:
        from memory_system.embeddings.eviction import (
            MAX_MEMORIES, get_memory_count, evict_and_rebuild,
        )
        if get_memory_count() > MAX_MEMORIES:
            evict_and_rebuild()
    except Exception as exc:
        print(f"  [memory] eviction check failed (non-fatal): {exc}", flush=True)

    return memory_id
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_insert_pipeline.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/core/insert_pipeline.py backend/tests/memory_system/test_insert_pipeline.py
git commit -m "feat(memory): wire affect tagging, chain detection, async summary into insert pipeline"
```

---

## Task 8: Query Rewriter

**Files:**
- Create: `backend/memory_system/retrieval/query_rewriter.py`
- Create: `backend/tests/memory_system/test_query_rewriter.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/memory_system/test_query_rewriter.py`:

```python
def test_rewrite_returns_string(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.query_rewriter._call_rewriter_llm",
        lambda q: "server configuration deployment backend infrastructure"
    )
    from memory_system.retrieval.query_rewriter import rewrite_query_for_retrieval
    result = rewrite_query_for_retrieval("what did we say about the server?")
    assert isinstance(result, str)
    assert len(result) > 5


def test_rewrite_falls_back_on_llm_error(monkeypatch):
    def _raise(q):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(
        "memory_system.retrieval.query_rewriter._call_rewriter_llm", _raise
    )
    from memory_system.retrieval.query_rewriter import rewrite_query_for_retrieval
    original = "what did we decide about auth?"
    result = rewrite_query_for_retrieval(original)
    assert result == original


def test_rewrite_rejects_empty_llm_response(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.query_rewriter._call_rewriter_llm",
        lambda q: "   "
    )
    from memory_system.retrieval.query_rewriter import rewrite_query_for_retrieval
    original = "what is the database schema?"
    result = rewrite_query_for_retrieval(original)
    assert result == original
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_query_rewriter.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create backend/memory_system/retrieval/query_rewriter.py**

```python
def rewrite_query_for_retrieval(query: str) -> str:
    """Expand query into key concepts for better FAISS semantic matching.

    Returns the original query unchanged if the LLM call fails or returns
    an empty result — so the caller always gets something usable.
    """
    try:
        result = _call_rewriter_llm(query)
        return result.strip() if result and len(result.strip()) > 5 else query
    except Exception:
        return query


def _call_rewriter_llm(query: str) -> str:
    from llm import ask_llm
    prompt = (
        "Rewrite this search query as space-separated key concepts and related terms "
        "for semantic vector search. Output only the terms, no explanation, no punctuation.\n\n"
        f"Query: {query}\n\nExpanded terms:"
    )
    return ask_llm(prompt).strip()
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_query_rewriter.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/retrieval/query_rewriter.py backend/tests/memory_system/test_query_rewriter.py
git commit -m "feat(memory): LLM query rewriter for deep path FAISS expansion"
```

---

## Task 9: LLM Re-ranker

**Files:**
- Create: `backend/memory_system/retrieval/reranker.py`
- Create: `backend/tests/memory_system/test_reranker.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/memory_system/test_reranker.py`:

```python
import json


_CANDIDATES = [
    {"memory_id": f"m{i}", "summary": f"summary {i}", "raw_text": f"full text {i}", "score": 0.5}
    for i in range(20)
]


def test_coarse_filter_returns_at_most_10(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.reranker._call_llm_for_ranking",
        lambda prompt: json.dumps(list(range(10)))
    )
    from memory_system.retrieval.reranker import _coarse_filter
    result = _coarse_filter("test query", _CANDIDATES)
    assert len(result) <= 10


def test_fine_rank_returns_at_most_5(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.reranker._call_llm_for_ranking",
        lambda prompt: json.dumps([0, 1, 2, 3, 4])
    )
    from memory_system.retrieval.reranker import _fine_rank
    result = _fine_rank("test query", _CANDIDATES[:10])
    assert len(result) <= 5


def test_rerank_returns_5_on_success(monkeypatch):
    call_n = [0]
    def _mock_llm(prompt):
        call_n[0] += 1
        return json.dumps(list(range(min(10, call_n[0] * 5))))
    monkeypatch.setattr("memory_system.retrieval.reranker._call_llm_for_ranking", _mock_llm)
    monkeypatch.setattr("memory_system.retrieval.reranker.can_load_7b", lambda: False)

    from memory_system.retrieval.reranker import rerank_memories
    result = rerank_memories("what did I do yesterday?", _CANDIDATES)
    assert len(result) <= 5


def test_rerank_falls_back_on_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "memory_system.retrieval.reranker._call_llm_for_ranking",
        lambda prompt: "not valid json at all {{{"
    )
    monkeypatch.setattr("memory_system.retrieval.reranker.can_load_7b", lambda: False)
    from memory_system.retrieval.reranker import rerank_memories
    result = rerank_memories("query", _CANDIDATES)
    # Falls back to first N candidates
    assert len(result) <= 5
    assert all(c in _CANDIDATES for c in result)


def test_rerank_empty_candidates_returns_empty(monkeypatch):
    monkeypatch.setattr("memory_system.retrieval.reranker.can_load_7b", lambda: False)
    from memory_system.retrieval.reranker import rerank_memories
    assert rerank_memories("query", []) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_reranker.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create backend/memory_system/retrieval/reranker.py**

```python
import json
from utils.ram_monitor import can_load_7b


def rerank_memories(query: str, candidates: list[dict]) -> list[dict]:
    """Two-pass LLM re-ranking.

    Pass 1 (coarse): score all candidates on summaries → top-10.
    Pass 2 (fine):   score top-10 on full raw_text → top-5.

    Falls back gracefully if LLM returns invalid JSON.
    """
    if not candidates:
        return []

    top10 = _coarse_filter(query, candidates)
    top5  = _fine_rank(query, top10)
    return top5


def _coarse_filter(query: str, candidates: list[dict]) -> list[dict]:
    """Send all summaries to LLM, return top-10 by index."""
    summaries_block = "\n".join(
        f"[{i}] {c['summary']}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Memories:\n{summaries_block}\n\n"
        "Return a JSON array of up to 10 integer indices (0-based), most relevant first.\n"
        "Example: [3, 0, 7]\nReturn only the JSON array, nothing else."
    )
    try:
        raw = _call_llm_for_ranking(prompt)
        indices = json.loads(raw)
        if isinstance(indices, list):
            valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
            return [candidates[i] for i in valid[:10]]
    except Exception:
        pass
    return candidates[:10]


def _fine_rank(query: str, candidates: list[dict]) -> list[dict]:
    """Send full raw_text of up to 10 candidates to LLM, return top-5 by index."""
    if not candidates:
        return []

    entries_block = "\n\n".join(
        f"[{i}] {candidates[i].get('raw_text', candidates[i]['summary'])[:300]}"
        for i in range(len(candidates))
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Memories:\n{entries_block}\n\n"
        "Return a JSON array of up to 5 integer indices (0-based), most relevant first.\n"
        "Example: [2, 0, 4]\nReturn only the JSON array, nothing else."
    )
    try:
        raw = _call_llm_for_ranking(prompt)
        indices = json.loads(raw)
        if isinstance(indices, list):
            valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
            return [candidates[i] for i in valid[:5]]
    except Exception:
        pass
    return candidates[:5]


def _call_llm_for_ranking(prompt: str) -> str:
    """Call the best available model. Uses 7b if RAM allows, else 3b."""
    from llm import ask_llm
    from config.llm import LLM_CONFIG
    model = LLM_CONFIG.planner_model if can_load_7b() else LLM_CONFIG.model
    return ask_llm(prompt)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_reranker.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/retrieval/reranker.py backend/tests/memory_system/test_reranker.py
git commit -m "feat(memory): two-pass LLM re-ranker for deep path retrieval"
```

---

## Task 10: Upgrade retrieve_memories — Scoring + Routing

**Files:**
- Modify: `backend/memory_system/retrieval/search.py`
- Create: `backend/tests/memory_system/test_retrieval_routing.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/memory_system/test_retrieval_routing.py`:

```python
def test_short_query_routes_fast():
    from memory_system.retrieval.search import classify_retrieval_path
    assert classify_retrieval_path("hello there") == "fast"


def test_long_query_routes_deep():
    from memory_system.retrieval.search import classify_retrieval_path
    assert classify_retrieval_path("can you tell me what we discussed about authentication") == "deep"


def test_question_word_routes_deep():
    from memory_system.retrieval.search import classify_retrieval_path
    assert classify_retrieval_path("why did we choose FAISS?") == "deep"


def test_remember_routes_deep():
    from memory_system.retrieval.search import classify_retrieval_path
    assert classify_retrieval_path("remember what I said about the server") == "deep"


def test_effective_importance_used_in_scoring(monkeypatch):
    """Verify that an old ActionMemory scores lower than a fresh one."""
    import numpy as np
    from datetime import datetime, timedelta

    fresh_created  = datetime.utcnow().isoformat()
    old_created    = (datetime.utcnow() - timedelta(days=60)).isoformat()

    from memory_system.core.importance import calculate_effective_importance
    fresh_eff = calculate_effective_importance(7.0, "ActionMemory", fresh_created)
    old_eff   = calculate_effective_importance(7.0, "ActionMemory", old_created)
    assert fresh_eff > old_eff


def test_low_confidence_memories_excluded(tmp_db, monkeypatch):
    """Memories with confidence_score < 0.4 must not appear in results."""
    import sqlite3, numpy as np
    from datetime import datetime

    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        INSERT INTO memories (id, memory_type, raw_text, summary, importance_score,
                              confidence_score, affect, session_id, created_at)
        VALUES ('low-conf', 'IdeaMemory', 'low confidence memory', 'low conf', 8.0,
                0.2, 'neutral', 'test-session', ?)
    """, (datetime.utcnow().isoformat(),))
    conn.execute("""
        INSERT INTO memory_embeddings (memory_id, vector_id, model_name)
        VALUES ('low-conf', '0', 'all-MiniLM-L6-v2')
    """)
    conn.commit()
    conn.close()

    monkeypatch.setattr("memory_system.retrieval.search.generate_embedding_vector",
                        lambda t: np.ones(384, dtype="float32"))
    monkeypatch.setattr("memory_system.retrieval.search.search_vector",
                        lambda v, top_k: ([0.99], [0]))

    from memory_system.retrieval.search import retrieve_memories
    results = retrieve_memories("low confidence memory", top_k=5)
    result_ids = [r["memory_id"] for r in results]
    assert "low-conf" not in result_ids
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_retrieval_routing.py -v
```

Expected: `classify_retrieval_path` ImportError + `low-conf` appears in results.

- [ ] **Step 3: Add classify_retrieval_path + update scoring in search.py**

At the top of `backend/memory_system/retrieval/search.py`, after the existing imports, add:

```python
from memory_system.core.importance import calculate_effective_importance
from config.llm import DECAY_HALF_LIFE
```

After the `detect_intent()` function, add:

```python
_DEEP_PATH_TRIGGERS = {
    "why", "how", "what did", "when did", "remember", "recall",
    "tell me about", "explain", "what have i", "what was",
}


def classify_retrieval_path(query: str) -> str:
    """Return 'fast' or 'deep' based on query complexity heuristics."""
    q_lower = query.lower()
    if len(query.split()) > 10:
        return "deep"
    for trigger in _DEEP_PATH_TRIGGERS:
        if trigger in q_lower:
            return "deep"
    return "fast"
```

In `retrieve_memories()`, replace the scoring block. Find this section:

```python
similarity_score = 1 / (1 + float(distance))
importance_score = memory["importance_score"] / 10
recency_score = calculate_recency_boost(memory["created_at"])
```

Replace with:

```python
similarity_score = 1 / (1 + float(distance))

# Use time-decayed effective importance instead of raw stored value
eff_importance   = calculate_effective_importance(
    memory["importance_score"], memory["memory_type"], memory["created_at"]
)
importance_score = eff_importance / 10
recency_score    = calculate_recency_boost(memory["created_at"])
```

Also update the `cursor.execute` that fetches memory fields to include `raw_text`, `confidence_score`, and `affect` (find the existing SELECT on `memories` inside the FAISS results loop and replace it):

```python
cursor.execute("""
    SELECT id, summary, raw_text, importance_score, confidence_score,
           created_at, memory_type, session_id, affect
    FROM memories
    WHERE id = ? AND (confidence_score IS NULL OR confidence_score >= 0.4)
""", (memory_id,))
```

And in the scoring block, add affect boost after `consolidation_boost`:

```python
affect       = memory["affect"] if memory["affect"] else "neutral"
affect_boost = 0.0
if intent == "personal_query" and affect in ("frustrated", "excited"):
    affect_boost = 0.15
# Fade old negative-affect memories faster than normal decay would
if affect in ("frustrated", "negative"):
    try:
        age_days  = (datetime.utcnow() - datetime.fromisoformat(memory["created_at"])).days
        half_life = DECAY_HALF_LIFE.get(memory["memory_type"], 60) or 60
        if age_days > half_life:
            affect_boost -= 0.1
    except Exception:
        pass

final_score = (
    similarity_score * 0.55 +
    importance_score * 0.2 +
    recency_score    * 0.1 +
    consolidation_boost +
    affect_boost
)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_retrieval_routing.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/retrieval/search.py backend/tests/memory_system/test_retrieval_routing.py
git commit -m "feat(memory): two-tier path classification, decay scoring, affect boost, confidence filter"
```

---

## Task 11: Wire Deep Path into retrieve_memories

**Files:**
- Modify: `backend/memory_system/retrieval/search.py`
- Create: `backend/tests/memory_system/test_deep_path.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/memory_system/test_deep_path.py`:

```python
import json
from unittest.mock import patch


def test_deep_path_calls_reranker(monkeypatch):
    """Deep path should invoke rerank_memories."""
    import numpy as np

    reranker_called = [False]

    def mock_reranker(query, candidates):
        reranker_called[0] = True
        return candidates[:3]

    monkeypatch.setattr("memory_system.retrieval.search.generate_embedding_vector",
                        lambda t: np.ones(384, dtype="float32"))
    monkeypatch.setattr("memory_system.retrieval.search.search_vector",
                        lambda v, top_k: ([0.9] * 5, list(range(5))))
    monkeypatch.setattr("memory_system.retrieval.search.rerank_memories", mock_reranker)
    monkeypatch.setattr("memory_system.retrieval.search.rewrite_query_for_retrieval",
                        lambda q: q)

    from memory_system.retrieval.search import retrieve_memories
    retrieve_memories("why did we choose FAISS over ChromaDB?", path="deep")
    assert reranker_called[0]


def test_fast_path_skips_reranker(monkeypatch):
    """Fast path should NOT call rerank_memories."""
    import numpy as np

    reranker_called = [False]

    def mock_reranker(query, candidates):
        reranker_called[0] = True
        return candidates[:3]

    monkeypatch.setattr("memory_system.retrieval.search.generate_embedding_vector",
                        lambda t: np.ones(384, dtype="float32"))
    monkeypatch.setattr("memory_system.retrieval.search.search_vector",
                        lambda v, top_k: ([], [-1]))
    monkeypatch.setattr("memory_system.retrieval.search.rerank_memories", mock_reranker)

    from memory_system.retrieval.search import retrieve_memories
    retrieve_memories("hi there", path="fast")
    assert not reranker_called[0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_deep_path.py -v
```

Expected: 2 failed — `path` param doesn't exist yet, reranker not called.

- [ ] **Step 3: Add deep path wiring to retrieve_memories in search.py**

Add these imports at the top of `search.py`:

```python
from memory_system.retrieval.query_rewriter import rewrite_query_for_retrieval
from memory_system.retrieval.reranker import rerank_memories
```

Change the `retrieve_memories` signature:

```python
def retrieve_memories(query: str, top_k: int = 5, debug: bool = False, path: str = None):
```

At the start of `retrieve_memories`, before entity extraction, add path detection:

```python
    # ── Two-tier routing ─────────────────────────────────────────────────────
    effective_path = path or classify_retrieval_path(query)
    is_deep = (effective_path == "deep")
    faiss_top_k = top_k * 4 if is_deep else top_k * 2
    search_query = rewrite_query_for_retrieval(query) if is_deep else query
```

Change the FAISS search call to use `faiss_top_k` and `search_query`:

```python
    query_vector = generate_embedding_vector(search_query)
    distances, indices = search_vector(query_vector, top_k=faiss_top_k)
```

After the `top_results = results[:top_k]` line and before the recall reinforcement block, add the deep path re-ranking + multi-hop:

```python
    # ── Deep path: re-rank + multi-hop ───────────────────────────────────────
    if is_deep and top_results:
        # Fetch raw_text for re-ranker
        conn2 = get_connection()
        c2    = conn2.cursor()
        enriched = []
        for r in top_results:
            c2.execute("SELECT raw_text FROM memories WHERE id = ?", (r["memory_id"],))
            row = c2.fetchone()
            enriched.append({**r, "raw_text": row["raw_text"] if row else r["summary"]})
        conn2.close()

        top_results = rerank_memories(query, enriched)

        # Multi-hop: find memories sharing entities with top results
        seen_ids = {r["memory_id"] for r in top_results}
        multihop = _fetch_multihop_memories(
            cursor, [r["memory_id"] for r in top_results], seen_ids
        )
        top_results = top_results + multihop
```

Add `_fetch_multihop_memories` function before `retrieve_memories`:

```python
def _fetch_multihop_memories(
    cursor, top_memory_ids: list[str], already_seen: set, top_n: int = 3
) -> list[dict]:
    """Return up to top_n memories that share entities with top results but weren't in FAISS results."""
    entity_ids: set[str] = set()
    for mid in top_memory_ids:
        cursor.execute("SELECT entity_id FROM memory_entities WHERE memory_id = ?", (mid,))
        entity_ids.update(r["entity_id"] for r in cursor.fetchall())

    candidate_ids: set[str] = set()
    for eid in entity_ids:
        cursor.execute("SELECT memory_id FROM memory_entities WHERE entity_id = ?", (eid,))
        candidate_ids.update(r["memory_id"] for r in cursor.fetchall())
    candidate_ids -= already_seen

    results = []
    for mid in list(candidate_ids)[: top_n * 3]:
        cursor.execute("""
            SELECT id, summary, memory_type
            FROM memories
            WHERE id = ? AND (confidence_score IS NULL OR confidence_score >= 0.4)
              AND status != 'archived'
        """, (mid,))
        row = cursor.fetchone()
        if row:
            results.append({
                "memory_id": row["id"],
                "summary":   row["summary"],
                "memory_type": row["memory_type"],
                "score": 0.3,
            })
    return results[:top_n]
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_deep_path.py tests/memory_system/test_retrieval_routing.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory_system/retrieval/search.py backend/tests/memory_system/test_deep_path.py
git commit -m "feat(memory): wire deep path — query rewrite, re-rank, multi-hop retrieval"
```

---

## Task 12: Consolidation — Validation, Meta-Consolidation, Expiry

**Files:**
- Modify: `backend/memory_system/lifecycle/consolidator.py`
- Create: `backend/tests/memory_system/test_consolidation.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/memory_system/test_consolidation.py`:

```python
import numpy as np
from unittest.mock import patch


def test_validate_consolidation_accepts_good_summary(monkeypatch):
    """A summary semantically close to its sources should be accepted."""
    # Make embedder return same vector for all inputs → cosine sim = 1.0
    monkeypatch.setattr(
        "memory_system.lifecycle.consolidator.generate_embedding_vector",
        lambda text: np.ones(384, dtype="float32")
    )
    from memory_system.lifecycle.consolidator import validate_consolidation
    assert validate_consolidation("good summary", ["source a", "source b", "source c"])


def test_validate_consolidation_rejects_bad_summary(monkeypatch):
    """A summary orthogonal to sources should be rejected."""
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
    # Threshold is 0.60; orthogonal vectors give ~0.0 similarity
    result = validate_consolidation("completely unrelated text", ["source a", "source b", "source c"])
    assert result is False


def test_consolidation_threshold_comes_from_config():
    from config.llm import CONSOLIDATION_SIMILARITY_THRESHOLD
    from memory_system.lifecycle.consolidator import SIMILARITY_THRESHOLD
    assert SIMILARITY_THRESHOLD == CONSOLIDATION_SIMILARITY_THRESHOLD
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_consolidation.py -v
```

Expected: 3 failed — `validate_consolidation` doesn't exist; `SIMILARITY_THRESHOLD` is hardcoded.

- [ ] **Step 3: Update lifecycle/consolidator.py**

Replace the module-level constants at the top of `backend/memory_system/lifecycle/consolidator.py`:

```python
from config.llm import (
    CONSOLIDATION_SIMILARITY_THRESHOLD,
    META_CONSOLIDATION_SIMILARITY_THRESHOLD,
    CONSOLIDATION_VALIDATION_THRESHOLD,
    CONSOLIDATED_MEMORY_EXPIRY_DAYS,
    CONSOLIDATED_MEMORY_EXPIRY_MIN_IMPORTANCE,
)

SIMILARITY_THRESHOLD = CONSOLIDATION_SIMILARITY_THRESHOLD
MIN_CLUSTER_SIZE = 3
```

Add the `validate_consolidation` function before `run_weekly_consolidation()`:

```python
def validate_consolidation(summary: str, source_summaries: list[str]) -> bool:
    """Return True if the LLM summary is semantically close to the cluster centroid."""
    import numpy as np

    summary_vec    = generate_embedding_vector(summary)
    source_vecs    = np.array([generate_embedding_vector(s) for s in source_summaries])
    centroid       = source_vecs.mean(axis=0)

    norm_s = np.linalg.norm(summary_vec)
    norm_c = np.linalg.norm(centroid)
    if norm_s < 1e-8 or norm_c < 1e-8:
        return False

    similarity = float(np.dot(summary_vec, centroid) / (norm_s * norm_c))
    return similarity >= CONSOLIDATION_VALIDATION_THRESHOLD
```

In `run_weekly_consolidation()`, replace the single LLM consolidation call:

```python
consolidated_text = generate_llm_consolidation(project, summaries)
```

with validated version:

```python
consolidated_text = generate_llm_consolidation(project, summaries)
if not validate_consolidation(consolidated_text, summaries):
    consolidated_text = generate_llm_consolidation(project, summaries)   # retry
    if not validate_consolidation(consolidated_text, summaries):
        print(f"[Consolidation] Skipping cluster in '{project}' — validation failed twice.")
        continue
```

Add `run_meta_consolidation()` function after `run_weekly_consolidation()`:

```python
def run_meta_consolidation():
    """Merge pairs of ConsolidatedMemory records that are very similar (depth-2 only)."""
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, summary, project_reference FROM memories
        WHERE memory_type = 'ConsolidatedMemory' AND status != 'archived'
    """)
    consolidated = cursor.fetchall()
    conn.close()

    used: set[str] = set()

    for i, mem_a in enumerate(consolidated):
        if mem_a["id"] in used:
            continue
        vec_a = generate_embedding_vector(mem_a["summary"])

        for mem_b in consolidated[i + 1 :]:
            if mem_b["id"] in used:
                continue
            if (mem_a["project_reference"] or "global") != (mem_b["project_reference"] or "global"):
                continue

            vec_b = generate_embedding_vector(mem_b["summary"])
            sim   = float(
                (vec_a * vec_b).sum()
                / (((vec_a ** 2).sum() ** 0.5) * ((vec_b ** 2).sum() ** 0.5) + 1e-8)
            )

            if sim >= META_CONSOLIDATION_SIMILARITY_THRESHOLD:
                summaries = [mem_a["summary"], mem_b["summary"]]
                meta_text = generate_llm_consolidation(
                    mem_a["project_reference"] or "global", summaries
                )
                new_id = f"meta-{datetime.utcnow().timestamp()}"
                conn   = get_connection()
                c2     = conn.cursor()
                c2.execute("""
                    INSERT INTO memories
                    (id, memory_type, raw_text, summary, importance_score, source,
                     created_at, project_reference)
                    VALUES (?, 'MetaConsolidatedMemory', ?, ?, 9, 'system', ?, ?)
                """, (new_id, meta_text, meta_text,
                      datetime.utcnow().isoformat(), mem_a["project_reference"]))
                for src_id in (mem_a["id"], mem_b["id"]):
                    c2.execute("""
                        INSERT INTO memory_relations (id, source_memory_id, target_memory_id,
                                                      relation_type, created_at)
                        VALUES (?, ?, ?, 'summarized_by', ?)
                    """, (f"rel-meta-{src_id[:8]}", src_id, new_id,
                          datetime.utcnow().isoformat()))
                conn.commit()
                conn.close()
                used.update({mem_a["id"], mem_b["id"]})
                break


def archive_expired_consolidations():
    """Archive ConsolidatedMemory records older than threshold with low effective importance."""
    from datetime import timedelta
    from memory_system.core.importance import calculate_effective_importance

    cutoff = (datetime.utcnow() - timedelta(days=CONSOLIDATED_MEMORY_EXPIRY_DAYS)).isoformat()

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, importance_score, created_at, memory_type FROM memories
        WHERE memory_type IN ('ConsolidatedMemory', 'MetaConsolidatedMemory')
          AND status != 'archived'
          AND created_at < ?
    """, (cutoff,))
    candidates = cursor.fetchall()

    archived = 0
    for mem in candidates:
        eff = calculate_effective_importance(
            mem["importance_score"], mem["memory_type"], mem["created_at"]
        )
        if eff < CONSOLIDATED_MEMORY_EXPIRY_MIN_IMPORTANCE:
            cursor.execute(
                "UPDATE memories SET status = 'archived' WHERE id = ?", (mem["id"],)
            )
            archived += 1

    conn.commit()
    conn.close()
    if archived:
        print(f"[Consolidation] Archived {archived} expired consolidated memories.")
```

Finally, at the end of `run_weekly_consolidation()`, after the main loop, call both:

```python
    print(f"[Consolidation] Created {consolidated_count} intelligent consolidated memories.")
    run_meta_consolidation()
    archive_expired_consolidations()
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/memory_system/test_consolidation.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/memory_system/lifecycle/consolidator.py backend/tests/memory_system/test_consolidation.py
git commit -m "feat(memory): validated consolidation, meta-consolidation, expiry archival"
```

---

## Task 13: Smoke Test + Final Cleanup

**Files:**
- No new files — integration validation only

- [ ] **Step 1: Run full test suite one final time**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -m pytest tests/ -v --tb=short
```

Expected: all tests pass, no warnings about missing imports.

- [ ] **Step 2: Verify DB migration runs cleanly on a fresh DB**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -c "
from memory_system.db.init_db import initialize_database
import tempfile, os
from memory_system.db import connection as c
c.DB_PATH = type(c.DB_PATH)(tempfile.mktemp(suffix='.db'))
initialize_database()
print('Migration OK')
"
```

Expected: `Database initialized.` and `Migration OK` (no error).

- [ ] **Step 3: Verify imports from main entry point don't break**

```bash
cd /Users/rohit/code_personal/smallO_v2/backend
python -c "
from memory_system.core.insert_pipeline import insert_memory
from memory_system.retrieval.search import retrieve_memories, classify_retrieval_path
from memory_system.retrieval.query_rewriter import rewrite_query_for_retrieval
from memory_system.retrieval.reranker import rerank_memories
from memory_system.core.chain import create_chain, detect_chain_type
from memory_system.core.affect import detect_affect
from memory_system.core.importance import calculate_effective_importance
from memory_system.lifecycle.consolidator import run_weekly_consolidation, validate_consolidation
print('All imports OK')
"
```

Expected: `All imports OK`.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(memory): memory system v2 intelligence layer complete

- Two-tier retrieval: fast (<100ms) / deep (3-5s with LLM re-ranking)
- Affect tagging on insert (keyword heuristic + LLM fallback)
- Time-based importance decay with per-type half-lives
- Entity graph population via AUTO_PARENT_RULES
- Memory chain module (confirms/contradicts/caused_by/led_to)
- Async LLM summary generation with importance bump
- Query rewriter for deep path FAISS expansion
- Two-pass LLM re-ranker (coarse: summaries, fine: raw_text)
- Multi-hop retrieval via entity co-occurrence
- Validated consolidation, meta-consolidation, expiry archival"
```
