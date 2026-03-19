from memory_system.core.insert_pipeline import insert_memory


def seed_profile():
    print("Seeding default profile memory...")

    profile_data = [
        {
            "text": "User name is Rohit",
            "memory_type": "PersonalMemory",
            "project_reference": "UserProfile"
        },
        {
            "text": "User age is 20",
            "memory_type": "PersonalMemory",
            "project_reference": "UserProfile"
        },
        {
            "text": "User is interested in Artificial Intelligence",
            "memory_type": "PersonalMemory",
            "project_reference": "UserProfile"
        },
        {
            "text": "User is a BCA student",
            "memory_type": "PersonalMemory",
            "project_reference": "UserProfile"
        },
        {
            "text": "User is building Small O voice assistant",
            "memory_type": "PersonalMemory",
            "project_reference": "UserProfile"
        }
    ]

    for item in profile_data:
        insert_memory(item)

    print("Profile memory seeded successfully.")


if __name__ == "__main__":
    seed_profile()


'''
        how to reset the memory system

rm memory_system/db/memory.db
rm memory_system/embeddings/faiss.index
python -m memory_system.db.init_db


        how to add the necessary info

python -m memory_system.seed_profile

'''