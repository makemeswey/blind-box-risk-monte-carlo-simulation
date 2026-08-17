"""Rare and secret figure experiments (section 15).

One figure at 1/144 does more to the cost of a collection than the other twelve
combined, because completion waits on the *slowest* figure. These experiments
re-run a collection with the secret set to 1/72, 1/144, 1/288, 1/500 and with no
secret at all, and report what happens to the mean, the tail, and the chance of
finishing inside a budget.

Rescaling keeps the distribution valid: the secret is pinned at the requested
probability and the regular figures absorb the remainder in proportion to their
existing weights.

    python -m src.simulation.experiments --collection JJK --budget 5000
    python -m src.simulation.experiments --collection JJK --analytical-only
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.database.connection import apply_schema, get_connection
from src.probability import coupon_collector as cc
from src.probability.distributions import Collection, load_collection, load_collections
from src.simulation import metrics
from src.simulation.simulator import save_run, simulate

# Secret rarities to sweep, as 1-in-N.
SECRET_DENOMINATORS = (72, 144, 288, 500)

DEFAULT_SIMULATIONS = 20_000
DEFAULT_SEED = 42


def with_secret_probability(collection: Collection, probability: float) -> Collection:
    """Pin every secret figure at `probability`, rescaling the rest to sum to 1."""
    secrets = collection.secret_indices
    if not secrets:
        raise ValueError(f"{collection.collection_id} has no secret figure")

    total_secret = probability * len(secrets)
    if not 0 < total_secret < 1:
        raise ValueError(f"Secret probability {probability} leaves no room for the rest")

    probabilities = collection.probabilities.copy()
    regulars = [i for i in range(collection.num_figures) if i not in secrets]
    probabilities[regulars] *= (1 - total_secret) / probabilities[regulars].sum()
    probabilities[list(secrets)] = probability
    return collection.replace(probabilities=probabilities)


def without_secrets(collection: Collection) -> Collection:
    """Drop the secret figures entirely — the "regulars only" baseline."""
    keep = [
        i for i in range(collection.num_figures) if i not in collection.secret_indices
    ]
    if not keep:
        raise ValueError("Collection is nothing but secrets")

    probabilities = collection.probabilities[keep]
    return collection.replace(
        figure_ids=tuple(collection.figure_ids[i] for i in keep),
        figure_names=tuple(collection.figure_names[i] for i in keep),
        rarities=tuple(collection.rarities[i] for i in keep),
        probabilities=probabilities / probabilities.sum(),
    )


def secret_scenarios(
    collection: Collection, denominators=SECRET_DENOMINATORS
) -> list[tuple[str, Collection]]:
    """(label, collection) for each secret rarity, plus the two baselines."""
    scenarios = [("no secret", without_secrets(collection))]
    scenarios += [
        (f"secret 1/{d}", with_secret_probability(collection, 1 / d))
        for d in denominators
    ]
    scenarios.append(("as published", collection))
    return scenarios


def _analytical_row(label: str, variant: Collection, budget: float | None) -> dict:
    summary = cc.analytical_summary(variant, budget)
    secrets = variant.secret_indices
    secret_probability = (
        float(variant.probabilities[secrets[0]]) if secrets else float("nan")
    )
    return {
        "scenario": label,
        "num_figures": variant.num_figures,
        "secret_probability": secret_probability,
        "secret_odds_1_in": 1 / secret_probability if secrets else float("inf"),
        "expected_boxes": summary["expected_boxes"],
        "median_boxes": summary["median_boxes"],
        "p95_boxes": summary["p95_boxes"],
        "expected_cost": summary["expected_cost"],
        "p95_cost": summary["p95_cost"],
        "tail_ratio": summary["p95_cost"] / summary["expected_cost"],
        "completion_probability": summary.get("completion_probability", float("nan")),
    }


def _simulated_row(
    label: str,
    variant: Collection,
    budget: float | None,
    num_simulations: int,
    seed: int | None,
) -> tuple[dict, object]:
    secrets = variant.secret_indices
    run = simulate(
        variant,
        num_simulations,
        seed=seed,
        budget=budget,
        # Enough to reconstruct the variant distribution from the stored run.
        params={
            "experiment": "secret_rarity",
            "scenario": label,
            "num_figures": variant.num_figures,
            "secret_probability": float(variant.probabilities[secrets[0]])
            if secrets
            else None,
        },
        track_milestones=False,
    )
    summary = metrics.summarise(run, budget)
    row = {
        "sim_expected_boxes": summary["mean_boxes"],
        "sim_median_boxes": summary["median_boxes"],
        "sim_p95_boxes": summary["p95_boxes"],
        "sim_expected_cost": summary["mean_cost"],
        "sim_p95_cost": summary["p95_cost"],
        "sim_completion_probability": summary.get(
            "completion_probability", float("nan")
        ),
        "sim_budget_for_95pct": summary["budget_for_95pct"],
        "standard_error_boxes": summary["standard_error_boxes"],
    }
    return row, run


def secret_rarity_experiment(
    collection: Collection,
    denominators=SECRET_DENOMINATORS,
    budget: float | None = None,
    num_simulations: int = DEFAULT_SIMULATIONS,
    seed: int | None = DEFAULT_SEED,
    analytical_only: bool = False,
    conn=None,
) -> pd.DataFrame:
    """Sweep the secret's rarity and report cost and tail risk for each setting.

    The analytical columns are exact, so the simulated ones are a cross-check
    rather than the source of truth; `conn` additionally persists each
    simulated scenario as its own run.
    """
    rows = []
    for label, variant in secret_scenarios(collection, denominators):
        row = _analytical_row(label, variant, budget)

        if not analytical_only:
            simulated, run = _simulated_row(
                label, variant, budget, num_simulations, seed
            )
            row.update(simulated)
            if conn is not None:
                row["run_id"] = save_run(conn, run)

        rows.append(row)

    frame = pd.DataFrame(rows)
    # Everything relative to the collection as it is actually sold.
    published = frame.loc[frame["scenario"] == "as published", "expected_cost"].iloc[0]
    frame["cost_vs_published"] = frame["expected_cost"] / published
    return frame


ANALYTICAL_COLUMNS = [
    "scenario",
    "num_figures",
    "secret_odds_1_in",
    "expected_boxes",
    "median_boxes",
    "p95_boxes",
    "expected_cost",
    "p95_cost",
    "tail_ratio",
    "cost_vs_published",
    "completion_probability",
]

SIMULATED_COLUMNS = [
    "scenario",
    "expected_boxes",
    "sim_expected_boxes",
    "standard_error_boxes",
    "expected_cost",
    "sim_expected_cost",
    "p95_cost",
    "sim_p95_cost",
    "completion_probability",
    "sim_completion_probability",
]


def _format(frame: pd.DataFrame, collection: Collection, budget: float | None) -> str:
    currency = collection.currency
    analytical = frame[[c for c in ANALYTICAL_COLUMNS if c in frame]].round(
        {
            "secret_odds_1_in": 0,
            "expected_boxes": 1,
            "p95_boxes": 0,
            "expected_cost": 2,
            "p95_cost": 2,
            "tail_ratio": 2,
            "cost_vs_published": 2,
            "completion_probability": 4,
        }
    )

    lines = [
        f"\n{collection.collection_name} ({collection.collection_id}) — "
        f"secret rarity sweep, {currency} {collection.box_price:.2f} per box",
        f"  exact coupon collector results"
        + (f", budget {currency} {budget:,.2f}" if budget else ""),
        "",
        analytical.to_string(index=False),
    ]

    if "sim_expected_boxes" in frame:
        simulated = frame[[c for c in SIMULATED_COLUMNS if c in frame]].round(
            {
                "expected_boxes": 1,
                "sim_expected_boxes": 1,
                "standard_error_boxes": 2,
                "expected_cost": 2,
                "sim_expected_cost": 2,
                "p95_cost": 2,
                "sim_p95_cost": 2,
                "completion_probability": 4,
                "sim_completion_probability": 4,
            }
        )
        lines += ["", "  Monte Carlo cross-check", "", simulated.to_string(index=False)]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="all")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--denominators",
        type=int,
        nargs="+",
        default=list(SECRET_DENOMINATORS),
        help="secret rarities to sweep, as 1-in-N (default: 72 144 288 500)",
    )
    parser.add_argument(
        "--analytical-only", action="store_true", help="skip the Monte Carlo columns"
    )
    parser.add_argument(
        "--save", action="store_true", help="persist each simulated scenario"
    )
    args = parser.parse_args()

    conn = get_connection(read_only=not args.save)
    try:
        if args.collection.lower() == "all":
            collections = list(load_collections(conn).values())
        else:
            collections = [load_collection(args.collection, conn)]

        if args.save:
            apply_schema(conn)

        for collection in collections:
            frame = secret_rarity_experiment(
                collection,
                denominators=args.denominators,
                budget=args.budget,
                num_simulations=args.simulations,
                seed=args.seed,
                analytical_only=args.analytical_only,
                conn=conn if args.save else None,
            )
            print(_format(frame, collection, args.budget))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
