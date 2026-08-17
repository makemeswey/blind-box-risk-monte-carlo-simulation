from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from src.database.connection import DB_PATH, get_connection

SUM_TOLERANCE = 1e-9

@dataclass(frozen=True)
class Collection:
    collection_id: str
    collection_name: str
    figure_ids: tuple[str, ...]
    figure_names: tuple[str, ...]
    rarities: tuple[str, ...]
    probabilities: np.ndarray
    box_price: float
    currency: str = "MYR"
    _cumulative: np.ndarray = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        probabilities = np.asarray(self.probabilities, dtype=float)
        object.__setattr__(self, "probabilities", probabilities)

        lengths = {
            len(self.figure_ids),
            len(self.figure_names),
            len(self.rarities),
            len(probabilities),
        }
        if len(lengths) != 1:
            raise ValueError(f"Mismatched figure/probability lengths: {lengths}")
        if probabilities.size == 0:
            raise ValueError("Collection has no figures")
        if np.any(probabilities <= 0):
            raise ValueError("All probabilities must be > 0 to ever complete a set")
        total = probabilities.sum()
        if abs(total - 1) > SUM_TOLERANCE:
            raise ValueError(f"Probabilities sum to {total!r}, not 1")
        if self.box_price <= 0:
            raise ValueError(f"box_price must be > 0, got {self.box_price}")

        object.__setattr__(self, "_cumulative", np.cumsum(probabilities))

    @property
    def num_figures(self) -> int:
        return len(self.probabilities)

    @property
    def is_uniform(self) -> bool:
        return bool(np.allclose(self.probabilities, self.probabilities[0]))

    @property
    def secret_indices(self) -> tuple[int, ...]:
        return tuple(i for i, r in enumerate(self.rarities) if r == "secret")

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        return np.searchsorted(self._cumulative, rng.random(size))

    def with_probabilities(self, probabilities) -> "Collection":
        probabilities = np.asarray(probabilities, dtype=float)
        return self.replace(probabilities=probabilities / probabilities.sum())

    def with_box_price(self, box_price: float) -> "Collection":
        return self.replace(box_price=float(box_price))

    def replace(self, **changes) -> "Collection":
        fields = {
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "figure_ids": self.figure_ids,
            "figure_names": self.figure_names,
            "rarities": self.rarities,
            "probabilities": self.probabilities,
            "box_price": self.box_price,
            "currency": self.currency,
        }
        fields.update(changes)
        return Collection(**fields)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "figure_id": self.figure_ids,
                "figure_name": self.figure_names,
                "rarity": self.rarities,
                "probability": self.probabilities,
                "odds_1_in": 1 / self.probabilities,
            }
        )

    @classmethod
    def uniform(cls, num_figures: int, box_price: float = 1.0, collection_id: str = "UNIFORM",) -> "Collection":
        ids = tuple(f"F{i + 1:02d}" for i in range(num_figures))
        return cls(
            collection_id=collection_id,
            collection_name=f"Uniform ({num_figures} figures)",
            figure_ids=ids,
            figure_names=ids,
            rarities=("regular",) * num_figures,
            probabilities=np.full(num_figures, 1 / num_figures),
            box_price=box_price,
        )


def from_frame(df: pd.DataFrame) -> Collection:
    ids = df["collection_id"].unique()
    if len(ids) != 1:
        raise ValueError(f"Expected exactly one collection, got {list(ids)}")

    df = df.sort_values("figure_id")
    return Collection(
        collection_id=str(ids[0]),
        collection_name=str(df["collection_name"].iloc[0]),
        figure_ids=tuple(df["figure_id"]),
        figure_names=tuple(df["figure_name"]),
        rarities=tuple(df["rarity"]),
        probabilities=df["probability"].to_numpy(),
        box_price=float(df["box_price"].iloc[0]),
        currency=str(df["currency"].iloc[0]),
    )


def _read_figures(conn, collection_id: str | None) -> pd.DataFrame:
    query = "SELECT * FROM figures"
    params = []
    if collection_id is not None:
        query += " WHERE collection_id = ?"
        params.append(collection_id)
    return conn.execute(query + " ORDER BY collection_id, figure_id", params).df()


def load_collections(conn=None, db_path: Path = DB_PATH) -> dict[str, Collection]:
    owned = conn is None
    conn = conn or get_connection(db_path, read_only=True)
    try:
        figures = _read_figures(conn, None)
    finally:
        if owned:
            conn.close()

    if figures.empty:
        raise ValueError(f"No figures in {db_path}; run `python -m src.data.load` first")

    return {
        collection_id: from_frame(group)
        for collection_id, group in figures.groupby("collection_id")
    }


def load_collection(collection_id: str, conn=None, db_path: Path = DB_PATH) -> Collection:
    owned = conn is None
    conn = conn or get_connection(db_path, read_only=True)
    try:
        figures = _read_figures(conn, collection_id)
    finally:
        if owned:
            conn.close()

    if figures.empty:
        raise ValueError(f"Unknown collection_id {collection_id!r}")
    return from_frame(figures)


if __name__ == "__main__":
    for collection in load_collections().values():
        print(f"\n{collection.collection_name} ({collection.collection_id})")
        print(
            f"  {collection.num_figures} figures, "
            f"{collection.currency} {collection.box_price:.2f} per box, "
            f"uniform={collection.is_uniform}"
        )
        print(collection.to_frame().to_string(index=False))
