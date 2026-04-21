# Memory System Upgrade — Intelligence Layer (v2)

**Date:** 2026-04-19
**Branch:** upgrade-memory
**Approach:** Option B — Intelligence Layer

---

## Overview

Holistic upgrade to the existing memory system (SQLite + FAISS + spaCy). The foundation is preserved; an intelligence layer is added on top covering smarter capture on insert, two-tier retrieval with LLM re-ranking, importance decay, memory chains, entity graph activation, and improved consolidation.

No schema replacement. No new vector database. All changes are additive or in-place upgrades to existing modules.

---

## Section 1: Architecture & Two-Tier Retrieval

The core foundation (SQLite + FAISS + spaCy) is untouched. A new intent classifier routes each query to either the fast path or deep path.

```
Query
  │
  ▼
┌─────────────────────────────────────────┐
│           Intent Classifier             │  ← NEW (rule-based, zero latency)
│   simple query → FAST PATH              │
│   complex query → DEEP PATH             │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
 FAST PATH         DEEP PATH
 (unchanged)       ├── Query Rewriter (LLM 3b)      ← NEW
 FAISS top-5       ├── FAISS top-20
 <100ms            ├── Entity Graph BFS (depth-2)    ← ACTIVATED
                   ├── LLM Re-ranker (7b) → top-5    ← NEW
                   ├── Multi-hop entity expansion     ← NEW
                   └── ~3–5s total
```

**Fast path:** unchanged from current implementation. No regression, no added latency.

**Deep path triggers when:**
- Query is >10 words, OR
- Contains question words: why, how, what did, when did, remember, OR
- User says "think carefully" / "deep search" / similar

**Intent classifier:** pure rule-based pattern matching + word count. Zero LLM call. No latency overhead.

**Latency targets:**
| Path | Target |
|---|---|
| Fast path | <100ms |
| Deep path | 3–5s (quality over speed) |

---

## Section 2: Insert Pipeline Upgrades

Three additions to the insert pipeline. All non-blocking — the SQLite record is inserted immediately, enrichment happens asynchronously.

### 2a. Async LLM Summary Generation
Replace truncation summary (first 300 chars) with an LLM-generated single-sentence abstractive summary. The record is inserted with the truncated placeholder, then `summary` is updated ~1s later when the LLM responds. Uses 3b model. The conversation is never blocked.

### 2b. Sentiment / Affect Tagging
New column `affect TEXT` on the `memories` table.

Values: `positive`, `negative`, `neutral`, `frustrated`, `excited`, `uncertain`

Detection order:
1. Keyword heuristic (fast, no LLM) — covers most cases
2. LLM call only if heuristic returns no match (input had no recognizable affect keywords)

Used in retrieval: emotional memories boosted for personal queries; old negative-affect memories decay faster.

### 2c. Smarter Importance Scoring
Keep the existing keyword tier. Add a semantic second-pass: when the async LLM summary returns, if it contains high-signal phrases (user preference, explicit decision, named goal), importance gets a one-time bump of +1.0–2.0. Reuses the summary generation call — no extra LLM call.

**What stays the same:** entity extraction, FAISS insert, deduplication threshold (0.90), UUID generation. Pipeline shape is identical.

---

## Section 3: LLM Re-ranker + Query Rewriting (Deep Path Only)

### 3a. Query Rewriting (`retrieval/query_rewriter.py`)
Before FAISS search on deep path, rewrite the query to be semantically richer.

```
User: "what did we say about the server?"
Rewritten: "server configuration backend infrastructure deployment discussion decisions"
```

- Model: qwen2.5:3b
- Prompt: ~50 tokens in, ~20 tokens out
- Latency: ~200ms
- The rewritten query is used for FAISS search only — original query shown to user

### 3b. LLM Re-ranker (`retrieval/reranker.py`)
After FAISS returns top-20 candidates (two sequential LLM calls):
1. **Call 1:** Send all 20 summaries to 7b model for coarse filter → top-10 (~1s)
2. **Call 2:** Send top-10 full `raw_text` to 7b model for final ranking → top-5 (~1–2s)

```
Input:  original query + 20 summaries + 10 full raw texts
Output: 5 ranked memory IDs + one-line reasoning per result
Model:  qwen2.5:7b (falls back to 3b if RAM < 3GB via can_load_7b())
Cost:   ~2–4s (two LLM calls total)
```

Reasoning is returned transiently (not persisted). Surfaced in debug mode.

### 3c. Multi-hop Retrieval
After re-ranking, extract entities from the top-5 memories. Search for additional memories sharing those entities that were not in the original FAISS top-20. Append relevant ones (up to 3) to the final context.

**Deep path total: ~3–5s**

---

## Section 4: Importance Dynamics — Decay + Affect

### 4a. Time-Based Importance Decay
Exponential decay computed lazily at retrieval time. The stored `importance_score` in SQLite is never modified by decay — only `effective_importance` used for ranking is affected. Decay is reversible and tunable.

```python
effective_importance = stored_importance × (0.5 ** (age_days / half_life_days))
```

Half-life per memory type (configurable in `config/llm.py`):
| Memory Type | Half-life |
|---|---|
| PersonalMemory | 180 days |
| DecisionMemory | 90 days |
| ActionMemory | 30 days |
| PlannerMemory | 14 days |
| ConsolidatedMemory | never decays |

### 4b. Affect in Retrieval Scoring
- `personal_query` intent + `frustrated`/`excited` affect → +0.15 score boost
- `knowledge_query` intent + `neutral` affect → no change
- Negative affect + age > half-life → additional 0.8× penalty (old frustrations fade)

### 4c. Confidence Scoring — Activated
`confidence_score` (currently always 1.0) now actually used.

- Starts at 1.0 on insert
- Decremented by 0.2 when a newer memory contradicts it (same entity, opposite sentiment)
- Incremented by 0.1 when a newer memory confirms it (capped at 1.0)
- Memories with `confidence_score < 0.4` excluded from retrieval unless explicitly requested

---

## Section 5: Memory Chains + Entity Graph Activation

### 5a. Entity Graph — Activated
The BFS expansion in `search.py` exists but never triggers because `entity_relations` is never populated. Fix: during entity upsert in the insert pipeline, write parent-child relations using `AUTO_PARENT_RULES` already defined in `entities/service.py` (e.g. FastAPI → Python, FAISS → VectorDB). Zero schema changes required.

On deep path: BFS expands from query entities to depth-2 neighbors, adding their linked memories to the FAISS candidate pool before re-ranking.

### 5b. Memory Chains (`core/chain.py` — new file)
Explicit causal links between memories. New `relation_type` values added to `memory_relations`:

| Relation | Meaning |
|---|---|
| `caused_by` | this decision happened because of that experience |
| `led_to` | this memory resulted in this outcome |
| `contradicts` | this memory conflicts with that one (feeds confidence decay) |
| `confirms` | corroborates an existing memory (boosts confidence) |

**Automatic chain creation:**
- During insert dedup check, if similarity is 0.75–0.90 (below 0.90 dedup threshold): link as `confirms` or `contradicts` based on affect polarity comparison

**Manual chain creation:**
- Planner can emit a `chain_memories` tool call to explicitly link a decision to its cause

**At retrieval:**
- Top-5 re-ranked memories with `caused_by` links: linked memories appended to context returned to LLM

---

## Section 6: Consolidation Improvements

### 6a. Semantic Validation Before Accepting
After LLM generates a consolidated summary:
1. Embed the summary
2. Compute cosine similarity against centroid of source memory cluster
3. If similarity > 0.60: accept
4. If not: retry once with tighter prompt
5. If second attempt fails: skip cluster this cycle, log warning

### 6b. Configurable Clustering Threshold
Move hardcoded `0.75` to `config/llm.py` as `CONSOLIDATION_SIMILARITY_THRESHOLD`.

### 6c. Re-consolidation (Meta-Consolidation)
If two `ConsolidatedMemory` records share >0.80 similarity AND same project: merge into `MetaConsolidatedMemory`. Depth capped at 2 levels. Handles long-running sessions where topics recur across weeks.

### 6d. Consolidated Memory Expiry
`ConsolidatedMemory` records older than 90 days with `effective_importance < 6` (after decay) are demoted to `archived`. Not deleted — excluded from retrieval. Prevents stale summaries from permanently occupying top retrieval slots.

---

## New Files

| File | Purpose |
|---|---|
| `retrieval/query_rewriter.py` | LLM query rewriting before FAISS search |
| `retrieval/reranker.py` | LLM re-ranking of FAISS candidates |
| `core/chain.py` | Memory chain creation and traversal |

## Modified Files

| File | Change |
|---|---|
| `db/schema.sql` + `db/init_db.py` | Add `affect TEXT` column to memories |
| `core/insert_pipeline.py` | Async summary, affect tagging, second-pass importance |
| `core/importance.py` | Semantic importance bump post-summary |
| `retrieval/search.py` | Two-tier routing, decay, affect scoring, confidence filter, multi-hop |
| `entities/service.py` | Populate `entity_relations` on insert |
| `lifecycle/consolidator.py` | Validation, re-consolidation, expiry |
| `config/llm.py` | Decay half-lives, consolidation threshold constants |

## No Changes

- `embeddings/vector_store.py` — FAISS index unchanged
- `embeddings/embedder.py` — model unchanged (all-MiniLM-L6-v2)
- `embeddings/eviction.py` — eviction logic unchanged
- WebSocket event types — no new events required
- `db/connection.py` — SQLite singleton unchanged

---

## Schema Changes

Single migration — add one column to `memories`:

```sql
ALTER TABLE memories ADD COLUMN affect TEXT DEFAULT 'neutral';
```

No other schema changes. All new relation types fit in the existing `memory_relations.relation_type TEXT` column.

---

## Constants to Add to `config/llm.py`

```python
# Memory decay half-lives (days)
DECAY_HALF_LIFE = {
    "PersonalMemory": 180,
    "DecisionMemory": 90,
    "ActionMemory": 30,
    "PlannerMemory": 14,
    "ConsolidatedMemory": None,  # never decays
}

# Consolidation
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.75
META_CONSOLIDATION_SIMILARITY_THRESHOLD = 0.80
CONSOLIDATION_VALIDATION_THRESHOLD = 0.60
CONSOLIDATED_MEMORY_EXPIRY_DAYS = 90
CONSOLIDATED_MEMORY_EXPIRY_MIN_IMPORTANCE = 6.0
```

---

## Risk Notes

- Async LLM summary update must handle the case where the memory is retrieved before the async update completes (return truncated summary, not an error)
- 7b re-ranker uses same RAM guard as planner (`can_load_7b()`) — automatic 3b fallback
- Entity graph BFS must be depth-limited (depth=2 max) to avoid runaway expansion on dense graphs
- Confidence decay must not trigger on semantically unrelated memories that share surface-level entity names
