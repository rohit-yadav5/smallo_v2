from memory_system.db.connection import get_connection
from memory_system.core.insert_pipeline import insert_memory
import os
from pathlib import Path


def reset_database():
    conn = get_connection()
    cursor = conn.cursor()

    # Clear all related tables
    cursor.execute("DELETE FROM memory_entities;")
    cursor.execute("DELETE FROM entity_relations;")
    cursor.execute("DELETE FROM entities;")
    cursor.execute("DELETE FROM memory_embeddings;")
    cursor.execute("DELETE FROM memories;")

    conn.commit()
    conn.close()

    # Remove FAISS index file
    index_path = Path("memory_system/embeddings/faiss.index")
    if index_path.exists():
        os.remove(index_path)

    print("Full database reset complete.")

def print_entities_and_relations():
    conn = get_connection()
    cursor = conn.cursor()

    print("\n--- ENTITIES ---")
    cursor.execute("SELECT id, name, domain, category, entity_type FROM entities;")
    for row in cursor.fetchall():
        print(dict(row))

    print("\n--- ENTITY RELATIONS ---")
    cursor.execute("""
        SELECT e1.name as child,
               e2.name as parent,
               r.relation_type
        FROM entity_relations r
        JOIN entities e1 ON r.source_entity_id = e1.id
        JOIN entities e2 ON r.target_entity_id = e2.id;
    """)
    for row in cursor.fetchall():
        print(dict(row))

    conn.close()


if __name__ == "__main__":

    reset_database()

    print("\nInserting MySQL test memory...\n")

    insert_memory({
        "text": "Implemented MySQL indexing optimization.",
        "memory_type": "ArchitectureMemory",
        "project_reference": "Small O"
    })

    print_entities_and_relations()