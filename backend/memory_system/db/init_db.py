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


if __name__ == "__main__":
    # Set reset=True during development
    initialize_database(reset=True)