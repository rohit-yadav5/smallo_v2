"""memory_system/embeddings/vector_store.py — FAISS index singleton.

Manages a module-level IndexFlatIP (brute-force cosine via inner product on
L2-normalised vectors).  Persists to faiss.index after every add.

replace_index() allows the eviction system to atomically swap the global
index reference after a rebuild without restarting the process.
"""

import faiss
import numpy as np
from pathlib import Path

VECTOR_DIM = 384
INDEX_PATH = Path(__file__).parent / "faiss.index"

# ---------------------------
# Load or Create Flat Index
# ---------------------------
if INDEX_PATH.exists():
    index = faiss.read_index(str(INDEX_PATH))
else:
    index = faiss.IndexFlatIP(VECTOR_DIM)  # safer on macOS


def normalize_vector(vector: np.ndarray):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def add_vector(memory_id: str, vector: np.ndarray):
    vector = normalize_vector(vector)
    index.add(np.array([vector]))
    faiss.write_index(index, str(INDEX_PATH))
    # Return FAISS position index (0-based sequential)
    return index.ntotal - 1


def search_vector(query_vector: np.ndarray, top_k: int = 5):
    query_vector = normalize_vector(query_vector)
    distances, indices = index.search(np.array([query_vector]), top_k)
    return distances[0], indices[0]


def replace_index(new_index: faiss.Index) -> None:
    """
    Atomically replace the module-level index reference.

    Called by the eviction system after a full FAISS rebuild so that all
    subsequent add_vector() and search_vector() calls use the new index
    without a process restart.  Thread-safety relies on the GIL; FAISS
    operations complete atomically with respect to Python object replacement.
    """
    global index
    index = new_index