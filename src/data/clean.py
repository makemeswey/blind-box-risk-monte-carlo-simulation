from __future__ import annotations
import re
from fractions import Fraction
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "collections.csv"
PROCESSED_PARQUET = PROJECT_ROOT / "data" / "processed" / "figures.parquet"

COLLECTION_NAMES = {
    "JJK": "Jujutsu Kaisen",
    "CRY": "Crybaby",
    "HIR": "Hirono",
}

RARITY_MAP = {
    "regular": "regular",
    "normal": "regular",
    "common": "regular",
    "rare": "rare",
    "secret": "secret",
    "chase": "secret",
}

CURRENCY_BY_SUFFIX = {
    "rm": "MYR",
    "myr": "MYR",
    "gbp": "GBP",
    "usd": "USD",
    "eur": "EUR",
}

DEFAULT_SOURCE = "Manually curated from public product listings (data/raw/collections.csv)"


RAW_SUM_TOLERANCE = 0.05


def parse_probability(value) -> float:
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    if text.endswith("%"):
        return float(text[:-1].strip()) / 100

    if "/" in text:
        numerator, _, denominator = text.partition("/")
        return float(Fraction(int(numerator.strip()), int(denominator.strip())))

    return float(text)


def parse_price(value) -> float:
    if pd.isna(value):
        return float("nan")

    text = re.sub(r"[^0-9.\-]", "", str(value))
    return float(text) if text else float("nan")


def parse_rarity(value) -> str:
    text = str(value).strip().lower()
    return RARITY_MAP.get(text, text)


def _find_price_column(columns) -> tuple[str, str]:
    for column in columns:
        match = re.fullmatch(r"box_price(?:_(\w+))?", column)
        if match:
            suffix = (match.group(1) or "").lower()
            return column, CURRENCY_BY_SUFFIX.get(suffix, "MYR")

    raise KeyError(f"No box price column found in {list(columns)}")


def normalise_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    totals = df.groupby("collection_id")["probability_raw"].transform("sum")
    df["probability"] = df["probability_raw"] / totals
    df["normalisation_factor"] = 1 / totals
    return df


def clean_collections(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    raw.columns = [column.strip().lower().replace(" ", "_") for column in raw.columns]

    price_column, currency = _find_price_column(raw.columns)

    df = pd.DataFrame(
        {
            "figure_id": raw["figure_id"].astype(str).str.strip().str.upper(),
            "figure_name": raw["figure_name"].astype(str).str.strip(),
            "rarity": raw["rarity"].map(parse_rarity),
            "probability_raw": raw["probability"].map(parse_probability),
            "box_price": raw[price_column].map(parse_price),
        }
    )
    df["currency"] = currency

    df["collection_id"] = df["figure_id"].str.extract(r"^([A-Z]+)", expand=False)
    df["collection_name"] = df["collection_id"].map(COLLECTION_NAMES)
    df["collection_name"] = df["collection_name"].fillna(df["collection_id"])

    df["source"] = (
        raw["source"].astype(str).str.strip()
        if "source" in raw.columns
        else DEFAULT_SOURCE
    )

    df = df.drop_duplicates(subset=["collection_id", "figure_id"], keep="first")

    df = normalise_probabilities(df)

    columns = [
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
    return df[columns].sort_values(["collection_id", "figure_id"]).reset_index(drop=True)


def write_processed(df: pd.DataFrame, path: Path = PROCESSED_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    figures = clean_collections()
    print(figures.to_string(index=False))
    print(f"\nWrote {len(figures)} figures to {write_processed(figures)}")
