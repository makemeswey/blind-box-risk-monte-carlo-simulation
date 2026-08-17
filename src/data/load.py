from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
from src.data.clean import PROCESSED_PARQUET, RAW_CSV, clean_collections, write_processed
from src.data.validate import ERROR, summarise, validate
from src.database.connection import DB_PATH, apply_schema, get_connection

FIGURE_COLUMNS = [
    "figure_id",
    "collection_id",
    "collection_name",
    "figure_name",
    "rarity",
    "probability_raw",
    "probability",
    "normalisation_factor",
    "box_price",
    "currency",
    "source",
]


def load_figures(conn, figures: pd.DataFrame) -> int:
    payload = figures[FIGURE_COLUMNS]
    collection_ids = payload["collection_id"].unique().tolist()

    conn.register("incoming_figures", payload)
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            "DELETE FROM figures WHERE collection_id IN "
            "(SELECT UNNEST($1::VARCHAR[]))",
            [collection_ids],
        )
        conn.execute(
            f"INSERT INTO figures ({', '.join(FIGURE_COLUMNS)}) "
            f"SELECT {', '.join(FIGURE_COLUMNS)} FROM incoming_figures"
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister("incoming_figures")

    return len(payload)


def run_etl(raw_csv: Path = RAW_CSV, db_path: Path = DB_PATH, parquet_path: Path = PROCESSED_PARQUET, allow_invalid: bool = False,) -> pd.DataFrame:
    figures = clean_collections(raw_csv)
    figures, issues = validate(figures)

    print(summarise(issues))

    has_errors = not issues.empty and (issues["level"] == ERROR).any()
    if has_errors and not allow_invalid:
        raise ValueError(
            "Validation errors found; fix the raw data or rerun with --allow-invalid"
        )

    valid = figures[figures["is_valid"]].drop(columns=["is_valid"])
    if valid.empty:
        raise ValueError("No valid figures to load")

    write_processed(valid, parquet_path)

    conn = get_connection(db_path)
    try:
        apply_schema(conn)
        loaded = load_figures(conn, valid)
        summary = conn.execute(
            """
            SELECT collection_id,
                   any_value(collection_name)  AS collection_name,
                   count(*)                    AS num_figures,
                   sum(probability)            AS total_probability,
                   any_value(box_price)        AS box_price
            FROM figures
            GROUP BY collection_id
            ORDER BY collection_id
            """
        ).df()
    finally:
        conn.close()

    print(f"\nWrote {parquet_path}")
    print(f"Loaded {loaded} figures into {db_path}\n")
    print(summary.to_string(index=False))
    return valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=RAW_CSV)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="load the valid rows even when validation reports errors",
    )
    args = parser.parse_args()

    try:
        run_etl(args.raw_csv, args.db_path, allow_invalid=args.allow_invalid)
    except ValueError as error:
        print(f"\nETL aborted: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
