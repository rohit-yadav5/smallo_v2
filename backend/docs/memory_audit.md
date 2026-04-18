# Memory System Audit
_Generated: 2026-04-14 | Small O v2 backend_

---

## 1. Directory Structure

```
backend/memory_system/
├── schema.sql                    # SQLite DDL — 7 tables, 12 indexes
├── seed_profile.py               # One-time loader of 30 personal/project seed memories
│
├── db/
│   ├── connection.py             # get_connection() — opens SQLite with row_factory + FK enforcement
│   └── init_db.py                # run_schema() — executes schema.sql; optional reset=True wipes all tables
│
├── core/
│   ├── models.py                 # EMPTY — placeholder file, no code
│   ├── insert_pipeline.py        # Full write path: dedup → entity extract → importance → SQL insert → FAISS add
│   └── importance.py             # Rule-based importance scorer (flat rules, 5.0 baseline)
│
├── entities/
│   ├── extractor.py              # Calls ask_llm() to extract named entities from memory text (LLM-based)
│   ├── service.py                # get_or_create_entity() with AUTO_PARENT_RULES dict for auto-linking
│   └── registry.py              # CONTROLLED_ENTITIES hardcoded dict — canonical aliases for known entities
│
├── retrieval/
│   └── search.py                 # retrieve_memories() — full read path: graph expand → intent detect → FAISS → score
│
├── embeddings/
│   ├── embedder.py               # Lazy-loaded all-MiniLM-L6-v2 wrapper; generate_embedding_vector() + encode_async()
│   └── vector_store.py           # FAISS IndexFlatIP singleton; add_vector(), search_similar(), load/save
│
├── lifecycle/
│   ├── consolidator.py           # run_weekly_consolidation() — clusters similar memories and summarises with LLM
│   └── worker.py                 # run_lifecycle_maintenance() — archives stale low-importance memories
│
├── analytics/
│   └── monitor.py                # generate_health_report() — counts, importance distribution, top entities
│
└── tests/
    ├── test_dedup.py
    ├── test_memory.py
    ├── test_retrieval.py
    ├── test_search.py
    ├── test_llm_extraction.py
    └── test_auto_parent.py
```

---

## 2. Data Model

### SQLite Tables (backend/memory_system/schema.sql)

**`memories`** — primary record for every stored memory
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK AUTOINCREMENT | |
| content | TEXT NOT NULL | Raw text of the memory |
| memory_type | TEXT NOT NULL | One of the 8 type strings (see §3) |
| importance_score | REAL DEFAULT 5.0 | Rule-assigned at insert; bumped on retrieval |
| created_at | TEXT | ISO-8601 timestamp |
| updated_at | TEXT | ISO-8601 timestamp |
| source | TEXT | Origin tag (e.g. "conversation", "seed") |
| is_consolidated | INTEGER DEFAULT 0 | 1 if this memory was produced by the consolidator |
| is_archived | INTEGER DEFAULT 0 | 1 if archived by lifecycle worker |
| consolidation_group | TEXT | UUID linking memories consolidated together |

**`entities`** — named entities extracted from memories
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK AUTOINCREMENT | |
| name | TEXT UNIQUE NOT NULL | Canonical lower-cased name |
| entity_type | TEXT | "person", "project", "technology", "concept", etc. |
| parent_entity_id | INTEGER FK → entities.id | Hierarchical: "redis" → "cache" |
| created_at | TEXT | |
| mention_count | INTEGER DEFAULT 0 | Incremented on each use |

**`entity_relations`** — typed edges between entities
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK AUTOINCREMENT | |
| source_entity_id | INTEGER FK → entities.id | |
| target_entity_id | INTEGER FK → entities.id | |
| relation_type | TEXT | "uses", "depends_on", "built_with", etc. |
| created_at | TEXT | |

**`memory_entities`** — many-to-many join: memories ↔ entities
| Column | Type | Notes |
|---|---|---|
| memory_id | INTEGER FK → memories.id | |
| entity_id | INTEGER FK → entities.id | |
| PRIMARY KEY | (memory_id, entity_id) | Composite |

**`memory_relations`** — typed edges between memories
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK AUTOINCREMENT | |
| source_memory_id | INTEGER FK → memories.id | |
| target_memory_id | INTEGER FK → memories.id | |
| relation_type | TEXT | "related_to", "supersedes", "supports", etc. |
| strength | REAL DEFAULT 1.0 | Edge weight (unused in current retrieval) |
| created_at | TEXT | |

**`memory_embeddings`** — links a memory row to its FAISS vector
| Column | Type | Notes |
|---|---|---|
| memory_id | INTEGER FK → memories.id | 1-to-1 |
| vector_id | TEXT NOT NULL | FAISS integer index stored as TEXT string |
| model_name | TEXT | "all-MiniLM-L6-v2" |
| created_at | TEXT | |

**`memory_lifecycle`** — audit log of stage transitions
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK AUTOINCREMENT | |
| memory_id | INTEGER FK → memories.id | |
| stage | TEXT | "raw", "consolidated", "archived" |
| transitioned_at | TEXT | |
| reason | TEXT | Human-readable reason string |

### Indexes (12 total)
```sql
idx_memories_type           ON memories(memory_type)
idx_memories_importance     ON memories(importance_score)
idx_memories_created        ON memories(created_at)
idx_memories_consolidated   ON memories(is_consolidated)
idx_memories_archived       ON memories(is_archived)
idx_entities_name           ON entities(name)
idx_entities_type           ON entities(entity_type)
idx_entity_relations_source ON entity_relations(source_entity_id)
idx_entity_relations_target ON entity_relations(target_entity_id)
idx_memory_entities_memory  ON memory_entities(memory_id)
idx_memory_entities_entity  ON memory_entities(entity_id)
idx_memory_embeddings_mem   ON memory_embeddings(memory_id)
```

### FAISS Index
- **File:** `backend/memory_system/embeddings/faiss.index` (created on first `add_vector()`)
- **Type:** `IndexFlatIP` — brute-force inner-product (cosine after L2-normalised vectors)
- **Dimension:** 384 (all-MiniLM-L6-v2 output size)
- **Persistence:** saved to disk after every `add_vector()` call (no batching)
- **ID mapping:** FAISS uses sequential integer positions (0, 1, 2, …). The integer position is stored as a string in `memory_embeddings.vector_id`. Reverse-lookup requires scanning `memory_embeddings` for the matching `vector_id` value.
- **No ID-to-label map:** If a memory is deleted from SQLite, its vector remains in the FAISS index forever (orphaned). There is no `remove_ids()` call anywhere in the codebase.

---

## 3. Memory Types

Defined as string literals throughout the codebase (no enum class). All stored in `memories.memory_type`:

| Type String | Typical Source | Description |
|---|---|---|
| `PersonalMemory` | seed_profile.py, conversation | User profile facts (name, location, preferences) |
| `ProjectMemory` | seed_profile.py, conversation | Active/past project facts |
| `ArchitectureMemory` | conversation, explicit | System design decisions and component descriptions |
| `DecisionMemory` | conversation | Choices made and rationale (importance +2 bonus) |
| `ReflectionMemory` | post-turn async | LLM-generated introspective summary of a conversation turn |
| `ActionMemory` | post-plugin async | Records of tool/plugin actions executed (importance +1 bonus) |
| `IdeaMemory` | conversation | Ideas, brainstorms, hypotheses |
| `ConsolidatedMemory` | lifecycle/consolidator.py | LLM-generated summary of a cluster of similar memories |

---

## 4. Write Pipeline

**Entry point:** `core/insert_pipeline.py → insert_memory(content, memory_type, source, ...)`

**Code path (sequential, synchronous):**

```
insert_memory(content, memory_type, source)
│
├─ 1. DEDUP CHECK
│     generate_embedding_vector(content)          # embedder.py sync call
│     vector_store.search_similar(vec, k=5)       # FAISS brute-force search
│     If any result has score ≥ 0.90 → ABORT, return existing memory_id
│
├─ 2. ENTITY EXTRACTION
│     entities/extractor.py → extract_entities(content)
│     └─ ask_llm(prompt)                          # ⚠ FULL OLLAMA LLM CALL (cold: ~5-7s)
│        Returns JSON list of {name, type} objects
│
├─ 3. IMPORTANCE SCORING
│     core/importance.py → calculate_importance(content, memory_type)
│     Pure rule-based; no LLM call. Returns float 1.0–10.0.
│
├─ 4. SQL INSERT
│     INSERT INTO memories (content, memory_type, importance_score, source, ...)
│     Returns new memory_id (INTEGER)
│
├─ 5. ENTITY LINKING
│     For each extracted entity:
│     ├─ entities/service.py → get_or_create_entity(name, type)
│     │   ├─ entities/registry.py → check CONTROLLED_ENTITIES aliases
│     │   ├─ SELECT or INSERT INTO entities
│     │   └─ Check AUTO_PARENT_RULES → link parent_entity_id if matched
│     └─ INSERT INTO memory_entities (memory_id, entity_id)
│
├─ 6. FAISS VECTOR ADD
│     vector_store.add_vector(vec)
│     ├─ faiss_index.add(vec)                     # appends at next integer position N
│     └─ faiss_index.write_index(path)            # writes to disk immediately
│
└─ 7. EMBEDDING RECORD
      INSERT INTO memory_embeddings (memory_id, vector_id=str(N), model_name)
      Returns memory_id
```

**Threading context:** `insert_memory()` is always called from a background thread via `asyncio.run_coroutine_threadsafe` → thread pool. The FAISS write and SQLite write are both synchronous and blocking within that thread. The LLM call in step 2 is also synchronous within the thread.

---

## 5. Read Pipeline

**Entry point:** `retrieval/search.py → retrieve_memories(query, top_k=5)`

**Code path (sequential, synchronous):**

```
retrieve_memories(query, top_k=5)
│
├─ 1. ENTITY GRAPH EXPANSION
│     Extract keywords from query (simple split, no LLM)
│     SELECT entities WHERE name LIKE keyword%
│     For each matched entity → SELECT memory_ids via memory_entities JOIN
│     Collect candidate set: {memory_id: entity_match_score}
│
├─ 2. INTENT DETECTION
│     Simple keyword scan of query string:
│     "error"|"bug"|"issue"         → intent="debugging"    (importance weight ×1.5)
│     "how"|"what"|"explain"        → intent="knowledge"    (consolidation boost ×1.2)
│     "plan"|"architecture"|"build" → intent="planning"     (recency weight reduced)
│     default                       → intent="general"
│
├─ 3. SEMANTIC SEARCH
│     generate_embedding_vector(query)             # embedder.py sync call
│     vector_store.search_similar(vec, k=top_k*3) # FAISS returns (scores, positions)
│     Map FAISS positions → memory_ids via memory_embeddings table (SELECT loop)
│     Merge with entity candidates; keep union
│
├─ 4. FETCH & SCORE
│     SELECT * FROM memories WHERE id IN (candidates) AND is_archived=0
│     For each memory, compute composite score:
│
│     recency_score   = exp(-days_old / 30)
│     similarity_score = FAISS inner-product score (0–1 after normalisation)
│     consolidation_boost = 0.1 if is_consolidated else 0.0
│
│     final_score = (similarity_score × 0.55)
│                 + (importance_score × 0.2)   [importance normalised /10]
│                 + (recency_score × 0.1)
│                 + consolidation_boost
│
│     Intent adjustments applied on top of above weights.
│
├─ 5. SORT & TRUNCATE
│     Sort by final_score DESC; return top_k results
│
└─ 6. RECALL REINFORCEMENT
      For each returned memory:
      UPDATE memories SET importance_score = MIN(importance_score + 0.1, 10.0)
      (Bumps importance so frequently-retrieved memories surface more easily)
```

**Threading context:** `retrieve_memories()` is called synchronously in `_build_memory_context()` inside `main.py`, which runs in the asyncio event loop via `await asyncio.get_event_loop().run_in_executor(None, retrieve_memories, query)` — i.e. it is always offloaded to the default thread pool, never blocking the event loop directly.

---

## 6. Entity Extraction Pipeline

**Entry point:** `entities/extractor.py → extract_entities(text)`

**Full flow:**

```
extract_entities(text)
│
├─ Build prompt:
│   "Extract named entities from this text. Return JSON array of
│    {name: string, type: string} objects. Types: person, project,
│    technology, concept, place, organization."
│
├─ ask_llm(prompt, model=LLM_CONFIG.model)
│   └─ Full Ollama HTTP call (POST /api/generate)
│      ⚠ If model not in VRAM → cold load: 2–7 seconds
│      ⚠ This call does NOT use keep_alive=0 by default
│      ⚠ Runs inside synchronous pipeline thread → blocks that thread
│
├─ Parse JSON response
│   └─ Fallback: regex scan for known entity names if JSON fails
│
└─ Filter through CONTROLLED_ENTITIES registry (entities/registry.py)
    Canonical aliases: {"small o" → "small_o_assistant", "faiss" → "faiss", ...}
    Known list is small (~10 entries); most entities pass through unmodified.
```

**AUTO_PARENT_RULES** (entities/service.py) — automatically assigns `parent_entity_id`:
```python
AUTO_PARENT_RULES = {
    "redis":      "cache",
    "postgresql": "database",
    "sqlite":     "database",
    "react":      "frontend",
    "fastapi":    "backend",
    "ollama":     "llm_runtime",
    # ~10 more entries
}
```
If an entity name matches a key, the parent entity is found/created and linked.

**Why this is slow:** `ask_llm()` is a synchronous blocking call to Ollama. If the model has been evicted from VRAM (keep_alive=0), this triggers a cold model load (2–7s) before entity extraction can proceed. Since entity extraction runs inside `insert_memory()`, which runs after every conversation turn in a background thread, a cold model will cause every memory write to take 5–8s total instead of <100ms.

---

## 7. Memory Lifecycle

### A. Recall Reinforcement (automatic, per-retrieval)
- On every `retrieve_memories()` call, each returned memory's `importance_score` is bumped +0.1 (capped at 10.0).
- No decay mechanism exists. Importance only goes up, never down (except via manual edits).

### B. Consolidation (manual trigger only)
**Entry point:** `lifecycle/consolidator.py → run_weekly_consolidation()`

```
run_weekly_consolidation()
│
├─ SELECT all non-archived memories
├─ Load embeddings for all memories
├─ Cluster with SIMILARITY_THRESHOLD = 0.75, MIN_CLUSTER_SIZE = 3
│   (simple pairwise cosine; no FAISS used here — direct numpy)
├─ For each qualifying cluster:
│   ├─ ask_llm(prompt="Summarise these related memories: ...")
│   ├─ INSERT new ConsolidatedMemory with LLM summary text
│   ├─ UPDATE source memories SET consolidation_group = cluster_uuid
│   └─ INSERT lifecycle records (stage="consolidated")
└─ ⚠ NOT scheduled anywhere — must be called manually
```

### C. Archiving (manual trigger only)
**Entry point:** `lifecycle/worker.py → run_lifecycle_maintenance()`

```
run_lifecycle_maintenance()
│
├─ Archive condition: created_at > 30 days ago AND importance_score < 4.0
├─ UPDATE memories SET is_archived = 1
├─ INSERT memory_lifecycle records (stage="archived", reason="stale+low_importance")
└─ ⚠ NOT scheduled anywhere — must be called manually
```

Neither `run_weekly_consolidation()` nor `run_lifecycle_maintenance()` is called from `main.py` or any startup hook. They exist as standalone callable functions but are never triggered automatically.

### D. No Deletion
There is no code path that deletes a memory row from SQLite or removes a vector from FAISS. Archiving sets a flag; consolidation adds rows. The database and FAISS index only grow.

---

## 8. Integration Points

These are the locations **outside** `memory_system/` that interact with the memory system:

### backend/main.py

**`_build_memory_context(query: str) → str`**
- Called before every LLM call.
- Runs `retrieve_memories(query, top_k=5)` in the default thread pool.
- Formats results as numbered bullet list prepended to the system prompt.
- If retrieval fails, logs the error and returns `""` (graceful degradation).

**`_extract_identity_facts(user_text, assistant_text) → None`**
- Async background task fired after every turn via `asyncio.ensure_future`.
- Calls `ask_llm()` to check whether the turn contains a personal fact worth storing.
- If yes, calls `insert_memory(fact, "PersonalMemory", source="conversation")`.

**`_store_reflection(user_text, assistant_text) → None`**
- Async background task fired after every turn.
- Calls `ask_llm()` to generate a one-sentence reflection on the turn.
- Calls `insert_memory(reflection, "ReflectionMemory", source="reflection")`.

**`_store_action_memory(action_description: str) → None`**
- Called after every plugin or tool execution.
- Calls `insert_memory(action_description, "ActionMemory", source="action")`.
- No LLM call for the memory content itself — the action description is passed in directly.

### backend/memory_system/seed_profile.py
- Standalone script, not imported by `main.py`.
- Contains 30 hardcoded `PersonalMemory` and `ProjectMemory` entries about Rohit.
- Run once manually to populate the database with baseline context.

### Nowhere else
- The planner (`planner.py`) does NOT call `retrieve_memories()`. Plan steps have no memory context.
- Web agent tools do NOT interact with the memory system.
- The plugin router does NOT interact with the memory system.

---

## 9. Current Problems

### P1 — Entity extraction LLM call causes 5–8s write latency (CRITICAL)
`extractor.py` calls `ask_llm()` synchronously from inside `insert_memory()`, which is called in a background thread after every turn. If the model has been evicted from VRAM (`keep_alive=0`), this triggers a cold model load. With the new three-tier keep_alive strategy (ACTIVE=120s), the model should remain warm during active conversation — but any gap longer than 120s will cause the next memory write to stall.

**Workaround applied:** Three-tier keep_alive keeps model warm 120s after last turn. This does not eliminate the root cause.

**Proper fix:** Replace `ask_llm()` in the extractor with a fast keyword/NER approach (spaCy, or a dedicated tiny model), or run entity extraction only during consolidation (not on every insert).

### P2 — Tool/plugin output is not stored in memory
`_store_action_memory()` only stores the action description string passed by the caller, not the full tool output (e.g., web search results, weather data, file contents). Retrieved memories will record "performed web search for X" but not what the search returned.

### P3 — FAISS orphaned vectors on memory archiving
When a memory is archived (`is_archived=1`), its FAISS vector is not removed. Over time the FAISS index grows unboundedly. Semantic searches return archived memories' vectors which then fail the `WHERE is_archived=0` SQL filter — wasted work, and memory pressure from growing index.

### P4 — FAISS index saved to disk on every single insert
`vector_store.add_vector()` calls `faiss_index.write_index(path)` after every addition. For batch inserts (e.g., seeding 30 memories), this writes the full index file 30 times. At larger scale this becomes a disk write bottleneck.

### P5 — No memory cap / unbounded growth
There is no maximum memory count, no eviction based on capacity, and no pruning of low-importance memories except archiving (which still keeps the rows). A long-running server will accumulate memories indefinitely.

### P6 — Recall reinforcement has no decay
Importance scores are bumped +0.1 on every retrieval but never decay. Frequently-asked questions will eventually push all their associated memories to importance 10.0, crowding out genuinely high-importance but less-frequently-retrieved memories.

### P7 — Consolidation and archiving are never triggered automatically
`run_weekly_consolidation()` and `run_lifecycle_maintenance()` exist but are never scheduled. The database accumulates raw memories without ever consolidating or archiving them, defeating the lifecycle design.

### P8 — models.py is empty
`core/models.py` exists as a placeholder but contains no code. Dataclasses or TypedDicts for memory objects were presumably intended here but never implemented. Memory data flows as raw `sqlite3.Row` dicts and plain strings throughout, making the pipeline brittle and hard to type-check.

### P9 — No async path in write pipeline
`insert_memory()` is entirely synchronous. When called from background threads it is fine, but it cannot be awaited. If the call site ever moves to the async event loop directly (without `run_in_executor`), it will block the loop for up to 8s during entity extraction.

### P10 — entity_relations table is populated but never read
`entity_relations` is created in the schema and entities are linked via `AUTO_PARENT_RULES`, but `retrieval/search.py` does not traverse `entity_relations` edges — only `memory_entities` (memory→entity membership) is used. The relation graph built during writes is invisible to queries.

---

## 10. What Is NOT Implemented

| Feature | Status |
|---|---|
| `core/models.py` dataclasses | File exists, is empty — no typed memory models |
| Scheduled lifecycle triggers | `run_weekly_consolidation()` and `run_lifecycle_maintenance()` must be called manually — no cron, no startup hook |
| FAISS vector deletion | No `remove_ids()` call anywhere; archived memories' vectors are never cleaned up |
| Memory decay / importance decay | Importance only increases via recall reinforcement; no time-based decay |
| Memory count cap | No maximum number of memories; no LRU or capacity-based eviction |
| Planner memory context | `planner.py` does not call `retrieve_memories()` — plan execution has no memory context |
| Tool output storage | Plugin/tool results are not stored in memory, only the action label |
| entity_relations traversal in retrieval | Schema supports typed entity edges; retrieval never reads them |
| Async insert pipeline | `insert_memory()` is fully synchronous; no `async def insert_memory()` version |
| FAISS approximate nearest-neighbour | Uses `IndexFlatIP` (exact brute-force); no `IndexIVFFlat` or `IndexHNSW` for faster large-scale search |
| Embedding model upgrades | Model is hardcoded to `all-MiniLM-L6-v2`; no config option to swap |
| Multi-user support | All memories share a single SQLite DB with no user_id column; system is single-tenant only |
| Memory edit / correction UI | No API endpoint to correct, delete, or manually edit a memory |
| Confidence scores on entities | `extract_entities()` returns entities but no confidence; all entities are inserted with equal weight |
| Batch insert optimisation | `insert_memory()` inserts one memory at a time; no bulk insert path for seeding or consolidation |
