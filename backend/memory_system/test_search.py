from memory_system.embeddings.embedder import generate_embedding_vector
from memory_system.embeddings.vector_store import search_vector
from memory_system.db.connection import get_connection

query = "When did I deploy Small O?"

vector = generate_embedding_vector(query)
distances, indices = search_vector(vector, top_k=3)

print("Vector matches:", indices)

conn = get_connection()
cursor = conn.cursor()

for idx in indices:
    if idx == -1:
        continue

    cursor.execute("""
        SELECT memory_id FROM memory_embeddings
        WHERE vector_id = ?
    """, (str(idx),))

    row = cursor.fetchone()

    if row:
        cursor.execute("""
            SELECT summary FROM memories
            WHERE id = ?
        """, (row["memory_id"],))

        memory = cursor.fetchone()
        print("Matched Memory:", memory["summary"])

conn.close()