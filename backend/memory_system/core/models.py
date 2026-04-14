"""memory_system/core/models.py — Typed data models for the memory pipeline.

Previously an empty placeholder.  All memory pipeline code should migrate toward
these types rather than passing raw sqlite3.Row dicts.  Callers that still use
raw dicts are marked with TODO(models) comments — migrate incrementally.

Conversion helper:  memory_from_row(row) → Memory
Return type:        retrieve_memories() returns list[MemorySearchResult]
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Memory:
    """A single stored memory record (mirrors the `memories` SQLite table)."""
    id: str                          # UUID / TEXT primary key
    content: str                     # the raw memory text (raw_text column)
    memory_type: str                 # "ReflectionMemory", "PersonalMemory", etc.
    importance: float                # 1.0–10.0; bumped on retrieval
    created_at: str                  # ISO-8601 timestamp
    source: str = "conversation"     # "conversation", "planner", "web", "seed", etc.
    last_accessed: Optional[str] = None   # ISO-8601 or None
    access_count: int = 0
    is_archived: bool = False
    is_consolidated: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class Entity:
    """A named entity extracted from one or more memories."""
    id: str
    name: str                        # lowercase canonical name
    entity_type: str                 # "Technology", "Service", "Concept", etc.
    domain: str                      # "Engineering", "Personal", etc.
    category: str                    # "Tool", "Database", etc.
    first_seen: str                  # ISO-8601
    mention_count: int = 1


@dataclass
class MemorySearchResult:
    """One item returned by retrieve_memories()."""
    memory_id: str
    summary: str                     # the memory text shown to the LLM
    memory_type: str
    score: float                     # composite retrieval score 0.0–1.0+
    rank: int = 0                    # 1 = most relevant


def memory_from_row(row) -> Memory:
    """
    Convert a sqlite3.Row or plain dict to a Memory dataclass.

    Handles both the old column name (raw_text) and a hypothetical future
    rename (content) gracefully.

    TODO(models): replace all raw dict access in insert_pipeline.py,
                  retrieval/search.py, and consolidator.py with this helper.
    """
    d = dict(row) if not isinstance(row, dict) else row
    # Support both column names — schema uses raw_text; models.py uses content
    content = d.get("content") or d.get("raw_text") or d.get("summary") or ""
    return Memory(
        id=str(d.get("id", "")),
        content=content,
        memory_type=d.get("memory_type", "ReflectionMemory"),
        importance=float(d.get("importance_score") or d.get("importance") or 5.0),
        created_at=d.get("created_at", ""),
        source=d.get("source", "conversation"),
        last_accessed=d.get("last_accessed") or d.get("updated_at"),
        access_count=int(d.get("access_count") or 0),
        is_archived=bool(d.get("status") == "archived" or d.get("is_archived")),
        is_consolidated=bool(d.get("memory_type") == "ConsolidatedMemory"
                             or d.get("is_consolidated")),
        metadata={},
    )


def search_result_from_dict(d: dict, rank: int = 0) -> MemorySearchResult:
    """
    Convert a raw result dict (from retrieve_memories) to a MemorySearchResult.

    TODO(models): update retrieve_memories() to return list[MemorySearchResult]
                  instead of list[dict].
    """
    return MemorySearchResult(
        memory_id=d.get("memory_id", ""),
        summary=d.get("summary", ""),
        memory_type=d.get("memory_type", ""),
        score=float(d.get("score", 0.0)),
        rank=rank,
    )
