"""backend/memory_system/retrieval/reranker.py — Two-pass LLM re-ranker.

Deep retrieval path: after semantic search returns candidates (typically 20+),
use the LLM to re-rank:

  Pass 1 (coarse): score all candidates on summaries → top-10.
  Pass 2 (fine):   score top-10 on full raw_text → top-5.

Falls back gracefully if LLM returns invalid JSON.
"""

import json
from utils.ram_monitor import can_load_7b


def rerank_memories(query: str, candidates: list[dict]) -> list[dict]:
    """Two-pass LLM re-ranking.

    Pass 1 (coarse): score all candidates on summaries → top-10.
    Pass 2 (fine):   score top-10 on full raw_text → top-5.

    Falls back gracefully if LLM returns invalid JSON.

    Args:
        query: The user's search query.
        candidates: List of candidate memory dicts with keys:
                   memory_id, summary, raw_text, score.

    Returns:
        List of up to 5 re-ranked memory dicts, most relevant first.
        Returns empty list if candidates is empty.
    """
    if not candidates:
        return []

    top10 = _coarse_filter(query, candidates)
    top5 = _fine_rank(query, top10)
    return top5


def _coarse_filter(query: str, candidates: list[dict]) -> list[dict]:
    """Send all summaries to LLM, return top-10 by index.

    Args:
        query: The user's search query.
        candidates: List of candidate memory dicts.

    Returns:
        List of up to 10 candidate dicts, ordered by LLM ranking.
        Falls back to first 10 if LLM response is invalid.
    """
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
    """Send full raw_text of up to 10 candidates to LLM, return top-5 by index.

    Args:
        query: The user's search query.
        candidates: List of up to 10 candidate memory dicts.

    Returns:
        List of up to 5 candidate dicts, ordered by LLM ranking.
        Falls back to first 5 if LLM response is invalid.
    """
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
    """Call the best available model for ranking.

    Uses 7b if RAM allows (>= 3GB free), otherwise falls back to 3b.

    Args:
        prompt: The ranking prompt (list of indices requested).

    Returns:
        The LLM's response (expected to be JSON, but may be invalid).
    """
    from llm import ask_llm
    return ask_llm(prompt)
