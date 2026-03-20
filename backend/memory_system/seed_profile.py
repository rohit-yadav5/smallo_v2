"""
seed_profile.py — Rohit's personal profile for Small O's memory system.

How to reset memory and re-seed:
    cd backend
    rm memory_system/db/memory.db
    rm memory_system/embeddings/faiss.index
    python -m memory_system.db.init_db
    python -m memory_system.seed_profile
"""

from memory_system.core.insert_pipeline import insert_memory


def seed_profile():
    print("Seeding Rohit's profile into memory...")

    profile_data = [

        # ── Identity ─────────────────────────────────────────────────────────
        {
            "text": "My name is Rohit. I am 20 years old and live in Gurugram, Haryana, India.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I prefer casual and direct communication. No need for formal language or corporate tone — just talk to me like a friend.",
            "memory_type": "PersonalMemory",
        },

        # ── Education ────────────────────────────────────────────────────────
        {
            "text": "I am in my 3rd and final year of BCA (Bachelor of Computer Applications) at Amity University, Gurugram.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I am graduating from Amity University Gurugram in 2026 with a BCA degree. This is my final semester.",
            "memory_type": "PersonalMemory",
        },

        # ── Work ─────────────────────────────────────────────────────────────
        {
            "text": "I am currently doing an internship at Sixhats. This is my primary work experience right now.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I need to find a new job or full-time role before May 2026. This is an urgent priority — it is only a few weeks away.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "My long-term career goal is to build a career in coding or the startup world. I am drawn to building products and companies, not just being an employee.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I want to explore career paths in AI, startup founding, or building developer tools. I am not fully sure of the direction yet but I am actively figuring it out.",
            "memory_type": "PersonalMemory",
        },

        # ── Projects ─────────────────────────────────────────────────────────
        {
            "text": "I am building Small O — a fully local, privacy-first voice AI assistant that runs entirely on my machine. It uses Whisper for STT, Ollama with phi3 for LLM, and Piper for TTS, connected to a cartoon-styled React frontend via WebSocket.",
            "memory_type": "ProjectMemory",
        },
        {
            "text": "Small O is my main personal AI project. I am both the developer and the primary user. Features I am still building include VAD (voice activity detection), speaker recognition, and remote deployment.",
            "memory_type": "ProjectMemory",
        },
        {
            "text": "I have another project called Second Earth. This is a separate project from Small O.",
            "memory_type": "ProjectMemory",
        },

        # ── Technical Skills ─────────────────────────────────────────────────
        {
            "text": "My core technical skills are in AI and machine learning — specifically NLP (natural language processing) and computer vision.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I work with Python as my primary language for AI/ML work. I use libraries like PyTorch, transformers, sentence-transformers, FAISS, OpenCV, and scikit-learn.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I build full-stack projects — I am comfortable with React, TypeScript, and Vite for frontend, and Python with asyncio and WebSockets for backend.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I use tools like Ollama for local LLM inference, Whisper for speech recognition, and Piper TTS for text-to-speech. I prefer local and open-source AI tools over cloud APIs.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I use Git and GitHub for version control. I work on a MacBook (macOS).",
            "memory_type": "PersonalMemory",
        },

        # ── Currently Learning ────────────────────────────────────────────────
        {
            "text": "I am currently deepening my knowledge in NLP — understanding how language models work at a fundamental level, including tokenisation, embeddings, attention mechanisms, and fine-tuning.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I am learning about LLM internals — how transformers work, how to run and fine-tune models locally, and how to build RAG (retrieval-augmented generation) pipelines.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I am building skills in audio processing and speech AI — including how voice activity detection works, how speaker diarisation works, and how to do real-time audio streaming from a browser to a Python backend.",
            "memory_type": "PersonalMemory",
        },

        # ── Hobbies & Interests ───────────────────────────────────────────────
        {
            "text": "I play badminton. It is one of my main physical activities and I enjoy it.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I watch anime. I enjoy it as a way to relax and unwind.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I play video games. Gaming is one of my hobbies alongside anime and badminton.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "My interests outside of coding are: badminton (sport), anime (entertainment), and gaming. I like switching between physical activity and entertainment to recharge.",
            "memory_type": "PersonalMemory",
        },

        # ── Lifestyle & Habits ───────────────────────────────────────────────
        {
            "text": "I am a night owl. I usually go to bed at around 3am. My most productive hours tend to be late at night.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I am actively trying to improve my health and diet. One of my current personal goals is to increase my protein intake. I am trying to eat better and be more consistent about it.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I am working on building healthier daily habits — better sleep schedule, better nutrition, more physical activity. These are ongoing personal improvement goals.",
            "memory_type": "PersonalMemory",
        },

        # ── Goals & Priorities ───────────────────────────────────────────────
        {
            "text": "My most urgent goal right now (as of March 2026) is to secure a new job or role before May 2026. I have about 6 weeks to make this happen.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "My short-term priorities are: (1) find a job before May, (2) graduate from Amity University, (3) keep building Small O and Second Earth.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "My long-term ambition is to find a career path at the intersection of coding and startups — whether that is joining an early-stage startup, founding something, or building indie products.",
            "memory_type": "PersonalMemory",
        },

        # ── How Small O should interact with Rohit ──────────────────────────
        {
            "text": "When talking to me, be casual and direct. Do not use overly formal language. Short, clear answers are preferred unless I ask for detail. Treat me like a smart friend who understands tech.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I am comfortable with technical depth — you can use proper technical terms around AI, ML, coding, and system design without over-explaining.",
            "memory_type": "PersonalMemory",
        },
        {
            "text": "I am a builder. When I bring up a problem, I generally want help thinking through a solution, not just a description of the problem. Be actionable.",
            "memory_type": "PersonalMemory",
        },

    ]

    for i, item in enumerate(profile_data, 1):
        print(f"  [{i}/{len(profile_data)}] Seeding: {item['text'][:60]}...")
        insert_memory(item)

    print(f"\nDone. {len(profile_data)} memories seeded successfully.")


if __name__ == "__main__":
    seed_profile()
