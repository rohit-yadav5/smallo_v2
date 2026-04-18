"""backend/memory_system/embeddings/embedder.py — Sentence embedding helper.

Wraps sentence-transformers/all-MiniLM-L6-v2 with lazy initialisation:
the model (~90 MB) is loaded on first call to generate_embedding_vector()
rather than at import time.  Once loaded it stays in RAM permanently —
it's small enough that evicting it between calls would waste more time
than it saves.

On a 16 GB Apple Silicon Mac this saves ~90 MB during the startup window
before any memory retrieval occurs (typically the first few seconds after
a server restart).

encode_async() offloads CPU-bound encoding to a dedicated thread pool so
callers running on the asyncio event loop don't block while the model runs.
Sync callers (e.g. retrieve_memories in its pipeline thread) can continue
using generate_embedding_vector() directly.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer

# Module-level singleton — None until first use, then stays loaded.
_model: SentenceTransformer | None = None

# Single-worker thread pool keeps encode calls serialised and off the event loop.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedder")


def _get_model() -> SentenceTransformer:
    """Return the embedding model, loading it on first call (lazy init)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_embedding_vector(text: str) -> np.ndarray:
    """Encode *text* synchronously and return a float32 numpy array.

    Safe to call from any thread.  Do NOT call from the asyncio event loop —
    use encode_async() there to avoid blocking.
    """
    embedding = _get_model().encode(text)
    return np.array(embedding).astype("float32")


async def encode_async(texts: list[str]) -> np.ndarray:
    """Encode a list of texts in the embedder thread pool (non-blocking).

    Offloads CPU-bound sentence-transformers work so the asyncio event loop
    is never stalled by model inference.  Falls back to synchronous encoding
    if there is no running event loop (e.g. called from a plain thread or
    test context).

    Returns a 2-D float32 array of shape (len(texts), 384).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop — fall back to synchronous encode (thread context).
        model = _get_model()
        return np.array(model.encode(texts, convert_to_numpy=True)).astype("float32")

    model = _get_model()  # lazy load; cached after first call
    result = await loop.run_in_executor(
        _executor,
        lambda: model.encode(texts, convert_to_numpy=True),
    )
    return np.array(result).astype("float32")
