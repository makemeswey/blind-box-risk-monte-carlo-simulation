"""Collection comparison (section 17): which set is the most expensive and risky?

Builds the side-by-side table over every collection in the database — box
price, figure count, expected and median boxes, expected and P95 cost,
duplicate rate, completion probability — and ranks them on cost and on risk.

Cost and risk are not the same question. Expected cost says what an average
collector pays; P95 cost and completion probability say what an unlucky one
pays, and those can rank differently when collections differ in rarity
structure rather than price.

    python -m src.analysis.comparison --budget 5000
    python -m src.analysis.comparison --budget 5000 --simulations 50000
    python -m src.analysis.comparison --budget 5000 --stored
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.database.connection import get_connection
from src.probability import coupon_collector as cc
from src.probability.distributions import Collection, load_collections
from src.simulation import metrics
from src.simulation.simulator import simulate

DEFAULT_SEED = 42

# Section 17's metric list, in display order.
METRIC_LABELS = {
    "box_price": "Box price",
    "num_figures": "Number of figures",
    "expected_boxes": "Expected boxes",
    "median_boxes": "Median boxes",
    "p95_boxes": "P95 boxes",
    "expected_cost": "Expected cost",
    "median_cost": "Median cost",
    "p95_cost": "P95 cost",
    "duplicate_rate": "Duplicate rate",
    "completion_probability": "Completion probability",
    "budget_for_95pct": "Budget for 95% confidence",
}


def _analytical_row(collection: Collection, budget: float | None) -> dict:
    summary = cc.analytical_summary(collection, budget)
    expected_boxes = summary["expected_boxes"]
    return {
        "collection_id": collection.collection_id,
        "collection_name": collection.collection_name,
        "currency": collection.currency,
        "box_price": collection.box_price,
        "num_figures": collection.num_figures,
        "num_secrets": len(collection.secret_indices),
        "expected_boxes": expected_boxes,
        "median_boxes": float(summary["median_boxes"]),
        "p95_boxes": float(summary["p95_boxes"]),
        "expected_cost": summary["expected_cost"],
        "median_cost": summary["median_boxes"] * collection.box_price,
        "p95_cost": summary["p95_cost"],
        # Ratio of expectations: E[duplicates]/E[boxes] = 1 - n/E[T].
        "duplicate_rate": 1 - collection.num_figures / expected_boxes,
        "completion_probability": summary.get("completion_probability", float("nan")),
        "budget_for_95pct": float(
            cc.boxes_for_quantile(collection.probabilities, 0.95) * collection.box_price
        ),
    }


def compare_collections(
    collections=None,
    budget: float | None = None,
    num_simulations: int = 0,
    seed: int | None = DEFAULT_SEED,
    conn=None,
) -> pd.DataFrame:
    """One row per collection, analytical by default.

    `num_simulations` adds Monte Carlo columns; the analytical ones stay the
    reference since they are exact.
    """
    if collections is None:
        collections = list(load_collections(conn).values())

    rows = []
    for collection in collections:
        row = _analytical_row(collection, budget)

        if num_simulations:
            run = simulate(
                collection, num_simulations, seed=seed, budget=budget,
                track_milestones=False,
            )
            summary = metrics.summarise(run, budget)
            row.update(
                {
                    "sim_expected_boxes": summary["mean_boxes"],
                    "sim_median_boxes": summary["median_boxes"],
                    "sim_expected_cost": summary["mean_cost"],
                    "sim_p95_cost": summary["p95_cost"],
                    "sim_duplicate_rate": summary["mean_duplicate_rate"],
                    "sim_completion_probability": summary.get(
                        "completion_probability", float("nan")
                    ),
                    "standard_error_boxes": summary["standard_error_boxes"],
                }
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("expected_cost", ascending=False).reset_index(
        drop=True
    )


def compare_stored_runs(conn, budget: float | None = None) -> pd.DataFrame:
    """The same comparison built from persisted runs rather than fresh simulation."""
    run_ids = [
        row[0]
        for row in conn.execute(
            """
            SELECT run_id FROM simulation_runs
            WHERE params IS NULL          -- baseline runs, not experiment variants
            QUALIFY row_number() OVER (
                PARTITION BY collection_id ORDER BY num_simulations DESC, run_id DESC
            ) = 1
            ORDER BY collection_id
            """
        ).fetchall()
    ]
    if not run_ids:
        raise ValueError("No baseline runs stored; run the simulator with --save first")

    return pd.DataFrame(
        [metrics.summarise_sql(conn, run_id, budget) for run_id in run_ids]
    ).sort_values("mean_cost", ascending=False).reset_index(drop=True)


def comparison_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Metrics as rows, collections as columns — section 17's table layout."""
    available = [metric for metric in METRIC_LABELS if metric in frame.columns]
    matrix = frame.set_index("collection_name")[available].T
    matrix.index = [METRIC_LABELS[metric] for metric in available]
    matrix.columns.name = None
    return matrix


def identical_structures(collections) -> bool:
    """True when every collection has the same probability distribution.

    Then any difference in the comparison comes from price alone — worth
    stating outright rather than implying the collections differ in risk.
    """
    reference = np.sort(collections[0].probabilities)
    return all(
        len(c.probabilities) == len(reference)
        and np.allclose(np.sort(c.probabilities), reference)
        for c in collections[1:]
    )


def risk_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank collections on each risk dimension (1 = worst for the collector)."""
    ranking = pd.DataFrame({"collection_name": frame["collection_name"]})
    ranking["by_expected_cost"] = frame["expected_cost"].rank(ascending=False, method="min").astype(int)
    ranking["by_p95_cost"] = frame["p95_cost"].rank(ascending=False, method="min").astype(int)
    ranking["by_expected_boxes"] = frame["expected_boxes"].rank(ascending=False, method="min").astype(int)
    if frame["completion_probability"].notna().any():
        ranking["by_completion_risk"] = (
            frame["completion_probability"].rank(ascending=True, method="min").astype(int)
        )
    return ranking


def verdict(frame: pd.DataFrame, collections=None) -> str:
    """One-paragraph answer to "which collection is riskiest to complete?"."""
    currency = frame["currency"].iloc[0] if "currency" in frame else ""
    costliest = frame.loc[frame["expected_cost"].idxmax()]
    tailiest = frame.loc[frame["p95_cost"].idxmax()]

    lines = [
        f"Most expensive on average: {costliest['collection_name']} "
        f"({currency} {costliest['expected_cost']:,.2f} expected, "
        f"{costliest['expected_boxes']:.0f} boxes)",
        f"Worst tail (P95 cost):     {tailiest['collection_name']} "
        f"({currency} {tailiest['p95_cost']:,.2f})",
    ]

    if frame["completion_probability"].notna().any():
        hardest = frame.loc[frame["completion_probability"].idxmin()]
        lines.append(
            f"Hardest inside budget:     {hardest['collection_name']} "
            f"({hardest['completion_probability']:.1%} completion probability)"
        )

    if collections and identical_structures(collections):
        lines.append(
            "\nNote: every collection here has the same probability structure "
            f"({collections[0].num_figures} figures, identical odds), so the box "
            "counts are identical and the ranking is driven purely by box price. "
            "The comparison only becomes a risk comparison once the collections "
            "differ in size or rarity."
        )
    return "\n".join(lines)


def _format_currency(matrix: pd.DataFrame, currency: str) -> pd.DataFrame:
    display = matrix.copy().astype(float).round(4)
    formatted = display.copy().astype(object)
    for label in display.index:
        for column in display.columns:
            value = display.loc[label, column]
            if "cost" in label.lower() or label == "Box price" or "Budget" in label:
                formatted.loc[label, column] = f"{currency} {value:,.2f}"
            elif "rate" in label.lower() or "probability" in label.lower():
                formatted.loc[label, column] = f"{value:.2%}"
            elif value == int(value):
                formatted.loc[label, column] = f"{int(value):,}"
            else:
                formatted.loc[label, column] = f"{value:,.1f}"
    return formatted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument(
        "--simulations",
        type=int,
        default=0,
        help="add Monte Carlo columns (0 = analytical only)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--stored", action="store_true", help="use persisted runs instead of simulating"
    )
    args = parser.parse_args()

    conn = get_connection(read_only=True)
    try:
        if args.stored:
            frame = compare_stored_runs(conn, args.budget)
            print(frame[
                [c for c in (
                    "run_id", "collection_id", "num_simulations", "box_price",
                    "mean_boxes", "median_boxes", "mean_cost", "p95_cost",
                    "mean_duplicate_rate", "completion_probability",
                ) if c in frame.columns]
            ].to_string(index=False))
            return 0

        collections = list(load_collections(conn).values())
        frame = compare_collections(
            collections,
            budget=args.budget,
            num_simulations=args.simulations,
            seed=args.seed,
        )

        currency = frame["currency"].iloc[0]
        print(f"\nCollection comparison" + (f" — budget {currency} {args.budget:,.2f}"
                                           if args.budget else ""))
        print(_format_currency(comparison_matrix(frame), currency).to_string())

        if args.simulations:
            simulated = frame[
                [c for c in frame.columns if c.startswith("sim_")]
                + ["collection_id", "standard_error_boxes"]
            ]
            print("\nMonte Carlo cross-check")
            print(simulated.round(3).to_string(index=False))

        print("\nRisk ranking (1 = worst for the collector)")
        print(risk_ranking(frame).to_string(index=False))
        print(f"\n{verdict(frame, collections)}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
