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

    # Return FAISS position index
    return index.ntotal - 1


def search_vector(query_vector: np.ndarray, top_k: int = 5):

    query_vector = normalize_vector(query_vector)

    distances, indices = index.search(np.array([query_vector]), top_k)
    return distances[0], indices[0]