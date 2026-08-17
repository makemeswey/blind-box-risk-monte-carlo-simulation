from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "database" / "popmart.duckdb"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection(db_path: Path = DB_PATH, read_only: bool = False):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(db_path, read_only=read_only)


def apply_schema(conn, schema_path: Path = SCHEMA_PATH) -> None:
    conn.execute(schema_path.read_text())


def create_connection():
    conn = get_connection()
    apply_schema(conn)
    conn.sql("SELECT 'Connected succesfully!'").show()
    conn.close()


if __name__ == "__main__":
    create_connection()
