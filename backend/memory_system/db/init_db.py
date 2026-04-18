from pathlib import Path
from .connection import get_connection, DB_PATH

SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"

def initialize_database(reset: bool = False):
    """
    Initialize the database schema.

    If reset=True, the existing database file will be deleted
    before recreating the schema. Use only during development.
    """

    # Ensure DB directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if reset and DB_PATH.exists():
        DB_PATH.unlink()
        print("Old database removed.")

    conn = get_connection()

    with open(SCHEMA_PATH, "r") as f:
        schema = f.read()

    conn.executescript(schema)
    conn.commit()
    conn.close()

    print("Database initialized.")
    migrate_database()


def migrate_database():
    """Apply incremental migrations to existing databases."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if affect column exists (SQLite has no IF NOT EXISTS for ALTER TABLE)
    cursor.execute("PRAGMA table_info(memories)")
    columns = {row["name"] for row in cursor.fetchall()}

    if "affect" not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN affect TEXT DEFAULT 'neutral'")
        conn.commit()
        print("Migration: added 'affect' column to memories.")

    conn.close()


if __name__ == "__main__":
    # Set reset=True during development
    initialize_database(reset=True)