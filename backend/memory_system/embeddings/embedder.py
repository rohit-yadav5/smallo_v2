from sentence_transformers import SentenceTransformer
import numpy as np

# Load once globally
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding_vector(text: str):
    embedding = model.encode(text)
    return np.array(embedding).astype("float32")