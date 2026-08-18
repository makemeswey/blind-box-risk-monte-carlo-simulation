"""Cached access to the database, the simulator and the analysis modules.

Every cache key is a plain value (collection id, simulation count, seed …) so
Streamlit can hash it; the `Collection` objects are rebuilt inside.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis import comparison, sensitivity
from src.database.connection import DB_PATH, get_connection
from src.probability import coupon_collector as cc
from src.probability.distributions import Collection, load_collections
from src.simulation import experiments, metrics
from src.simulation.simulator import SimulationRun, simulate

DEFAULT_SEED = 42

# Past ~50k runs the mean stops moving by more than a fraction of a box, so the
# slider stops there rather than spending seconds for no extra precision.
MIN_SIMULATIONS = 1_000
MAX_SIMULATIONS = 50_000
SIMULATION_STEP = 1_000
DEFAULT_SIMULATIONS = 10_000
SIMULATION_CHOICES = (1_000, 5_000, 10_000, 25_000, 50_000)


def clamp_simulations(num_simulations: int) -> int:
    """Keep a stored or typed count inside the supported range."""
    return int(min(max(int(num_simulations), MIN_SIMULATIONS), MAX_SIMULATIONS))


@st.cache_data(show_spinner=False)
def figures_frame(db_path: str = str(DB_PATH)) -> pd.DataFrame:
    """Every figure of every collection, straight from DuckDB."""
    conn = get_connection(Path(db_path), read_only=True)
    try:
        return conn.execute(
            """
            SELECT collection_id, collection_name, figure_id, figure_name, rarity,
                   probability_raw, probability, box_price, currency, source
            FROM figures
            ORDER BY collection_id, figure_id
            """
        ).df()
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def all_collections(db_path: str = str(DB_PATH)) -> dict[str, Collection]:
    return load_collections(db_path=Path(db_path))


def get_collection(collection_id: str) -> Collection:
    return all_collections()[collection_id]


def default_budget(collection: Collection) -> float:
    """Expected completion cost, rounded to a tidy number — a fair starting budget."""
    expected = cc.analytical_summary(collection)["expected_cost"]
    return float(round(expected / 100) * 100)


@st.cache_data(show_spinner=False)
def analytical_summary(collection_id: str, budget: float | None = None) -> dict:
    return cc.analytical_summary(get_collection(collection_id), budget)


@st.cache_data(show_spinner="Running simulations…", max_entries=8)
def run_simulation(
    collection_id: str,
    num_simulations: int,
    budget: float,
    seed: int = DEFAULT_SEED,
    track_milestones: bool = True,
) -> SimulationRun:
    return simulate(
        get_collection(collection_id),
        num_simulations,
        seed=seed,
        budget=budget,
        track_milestones=track_milestones,
    )


@st.cache_data(show_spinner="Comparing collections…", max_entries=8)
def comparison_frame(
    budget: float, num_simulations: int = 0, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    return comparison.compare_collections(
        list(all_collections().values()),
        budget=budget,
        num_simulations=num_simulations,
        seed=seed,
    )


@st.cache_data(show_spinner="Sweeping the secret rarity…", max_entries=8)
def secret_experiment(
    collection_id: str,
    denominators: tuple[int, ...],
    budget: float,
    num_simulations: int = 0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    return experiments.secret_rarity_experiment(
        get_collection(collection_id),
        denominators=denominators,
        budget=budget,
        num_simulations=num_simulations,
        seed=seed,
        analytical_only=not num_simulations,
    )


@st.cache_data(show_spinner="Running the sweep…", max_entries=16)
def sweep_frame(
    collection_id: str,
    parameter: str,
    values: tuple[float, ...],
    budget: float,
    num_simulations: int = 0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    return sensitivity.sweep(
        get_collection(collection_id),
        parameter,
        values=list(values),
        budget=budget,
        num_simulations=num_simulations,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def secret_change_impact(
    collection_id: str, budget: float, from_denominator: int, to_denominator: int
) -> dict:
    return sensitivity.secret_change_impact(
        get_collection(collection_id), budget, from_denominator, to_denominator
    )


def build_variant(
    collection: Collection,
    box_price: float | None = None,
    secret_denominator: int | None = None,
    num_figures: int | None = None,
) -> Collection:
    """A hypothetical version of a collection — resized, repriced, or re-rared."""
    variant = collection
    if num_figures is not None and num_figures != collection.num_figures:
        variant = sensitivity.resize_regulars(
            variant, int(num_figures) - len(variant.secret_indices)
        )
    if secret_denominator and variant.secret_indices:
        variant = experiments.with_secret_probability(variant, 1 / secret_denominator)
    if box_price:
        variant = variant.with_box_price(float(box_price))
    return variant


@st.cache_data(show_spinner="Evaluating the variant…", max_entries=32)
def variant_summary(
    collection_id: str,
    box_price: float,
    secret_denominator: int,
    num_figures: int,
    budget: float,
    num_simulations: int = 0,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Analytical summary of a modified collection, optionally cross-checked."""
    variant = build_variant(
        get_collection(collection_id), box_price, secret_denominator, num_figures
    )
    summary = cc.analytical_summary(variant, budget)

    if num_simulations:
        run = simulate(
            variant, num_simulations, seed=seed, budget=budget, track_milestones=False
        )
        simulated = metrics.summarise(run, budget)
        summary.update(
            {
                "sim_expected_boxes": simulated["mean_boxes"],
                "sim_expected_cost": simulated["mean_cost"],
                "sim_p95_cost": simulated["p95_cost"],
                "sim_completion_probability": simulated["completion_probability"],
            }
        )
    return summary
