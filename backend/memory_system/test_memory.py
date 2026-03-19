from memory_system.core.insert_pipeline import insert_memory

if __name__ == "__main__":

    for i in range(4):
        insert_memory({
            "text": f"Improved FAISS indexing performance iteration {i}",
            "memory_type": "ProjectMemory",
            "source": "manual",
            "project_reference": "Small O"
        })

    print("Inserted similar FAISS memories.")