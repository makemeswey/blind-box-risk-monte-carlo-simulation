"""Duplicate analysis (section 14): why finishing a collection gets so expensive.

The mechanism is simple. After a collector already owns the set S, the next box
is only useful with probability

    P(new) = 1 - sum_{i in S} p_i

so every acquisition lowers the chance the next box helps, and the expected
wait for the next new figure — 1 / P(new) — grows correspondingly. With equal
probabilities the k-th new figure costs n / (n - k + 1) boxes on average: the
1st is free, the last costs n boxes. A secret at 1/145 makes the final step
cost ~145 boxes on its own, which is why the duplicate rate climbs towards 100%
near completion.

Two views are produced:

  * `milestone_stats(run)` — per completion stage (1 figure, 2 figures, ...):
    boxes needed, marginal cost of that figure, duplicate rate at that point.
  * `unique_curve(...)`   — unique figures owned as a function of boxes opened,
    analytically E[U(t)] = sum_i (1 - (1 - p_i)^t), and from simulation.

    python -m src.analysis.duplicates --collection JJK --simulations 20000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.probability.distributions import Collection, load_collection, load_collections
from src.simulation.simulator import SimulationRun, simulate


def milestone_stats(run: SimulationRun) -> pd.DataFrame:
    """Per-stage view of a run: how the k-th new figure was earned.

    marginal_boxes is how many boxes were opened between the (k-1)-th and k-th
    distinct figure, and its reciprocal is the realised probability that a box
    contained something new at that stage.
    """
    milestones = run.require_milestones()
    n = run.collection.num_figures

    # Boxes spent getting from k-1 to k distinct figures.
    marginal = np.diff(milestones, axis=1, prepend=0)

    unique = np.arange(1, n + 1)
    boxes_at_stage = milestones.mean(axis=0)
    duplicates_at_stage = milestones - unique  # boxes opened minus figures owned

    return pd.DataFrame(
        {
            "unique_figures": unique,
            "completion_pct": unique / n,
            "mean_boxes": boxes_at_stage,
            "median_boxes": np.median(milestones, axis=0),
            "mean_marginal_boxes": marginal.mean(axis=0),
            "p95_marginal_boxes": np.percentile(marginal, 95, axis=0),
            "marginal_probability_new": 1 / marginal.mean(axis=0),
            "mean_duplicates": duplicates_at_stage.mean(axis=0),
            "duplicate_rate": (duplicates_at_stage / milestones).mean(axis=0),
            "mean_cost": boxes_at_stage * run.collection.box_price,
            "marginal_cost": marginal.mean(axis=0) * run.collection.box_price,
            # Equal-probability benchmark for the same stage.
            "uniform_marginal_boxes": n / (n - unique + 1),
        }
    )


def expected_unique(collection: Collection, boxes) -> np.ndarray:
    """E[unique figures owned] after `boxes` boxes — exact, no simulation.

    Figure i is missing after t boxes with probability (1 - p_i)^t, and
    expectations add even though the indicators are dependent.
    """
    boxes = np.asarray(boxes, dtype=float)
    missing = (1 - collection.probabilities) ** boxes[:, None]
    return (1 - missing).sum(axis=1)


def unique_curve(
    collection: Collection,
    max_boxes: int | None = None,
    num_points: int = 200,
    run: SimulationRun | None = None,
) -> pd.DataFrame:
    """Unique figures and duplicate rate as a function of boxes opened.

    The duplicate rate here is 1 - E[unique]/boxes, a ratio of expectations —
    close to, but not identical with, the average of per-simulation rates.
    """
    if max_boxes is None:
        max_boxes = int(np.percentile(run.boxes, 95)) if run else 500
    boxes = np.unique(np.linspace(1, max_boxes, num_points).astype(int))

    analytical = expected_unique(collection, boxes)
    curve = pd.DataFrame(
        {
            "boxes": boxes,
            "expected_unique": analytical,
            "expected_duplicates": boxes - analytical,
            "duplicate_rate": 1 - analytical / boxes,
            "completion_pct": analytical / collection.num_figures,
        }
    )

    if run is not None:
        # E[U(t)] = sum_k P(k-th figure arrived by t): count milestones <= t.
        milestones = run.require_milestones()
        reached = np.stack(
            [
                np.searchsorted(np.sort(milestones[:, k]), boxes, side="right")
                for k in range(collection.num_figures)
            ]
        )
        curve["simulated_unique"] = reached.sum(axis=0) / run.num_simulations
        curve["simulated_duplicate_rate"] = 1 - curve["simulated_unique"] / boxes
    return curve


def duplicate_summary(run: SimulationRun) -> dict:
    """Headline duplicate numbers for one run."""
    results = run.results
    stats = milestone_stats(run)
    n = run.collection.num_figures
    last_figure = stats.iloc[-1]

    return {
        "collection_id": run.collection.collection_id,
        "collection_name": run.collection.collection_name,
        "num_figures": n,
        "mean_boxes": float(run.boxes.mean()),
        "mean_duplicates": float(results["duplicates"].mean()),
        "mean_duplicate_rate": float(results["duplicate_rate"].mean()),
        "pooled_duplicate_rate": float(
            results["duplicates"].sum() / results["boxes_required"].sum()
        ),
        # The final figure alone: the coupon collector's sting.
        "final_figure_mean_boxes": float(last_figure["mean_marginal_boxes"]),
        "final_figure_mean_cost": float(last_figure["marginal_cost"]),
        "final_figure_share_of_boxes": float(
            last_figure["mean_marginal_boxes"] / run.boxes.mean()
        ),
        "boxes_for_half_the_set": float(stats.iloc[n // 2 - 1]["mean_boxes"]),
    }


def _format(run: SimulationRun) -> str:
    collection = run.collection
    stats = milestone_stats(run)
    summary = duplicate_summary(run)
    currency = collection.currency

    display = stats[
        [
            "unique_figures",
            "completion_pct",
            "mean_boxes",
            "mean_marginal_boxes",
            "uniform_marginal_boxes",
            "marginal_probability_new",
            "duplicate_rate",
            "marginal_cost",
        ]
    ].copy()
    display["completion_pct"] = (display["completion_pct"] * 100).round(1)
    display["duplicate_rate"] = (display["duplicate_rate"] * 100).round(1)
    display = display.round(
        {
            "mean_boxes": 1,
            "mean_marginal_boxes": 1,
            "uniform_marginal_boxes": 1,
            "marginal_probability_new": 4,
            "marginal_cost": 2,
        }
    )
    display.columns = [
        "unique",
        "complete%",
        "boxes",
        "marginal",
        "uniform_ref",
        "P(new)",
        "dup%",
        f"marginal_{currency}",
    ]

    return "\n".join(
        [
            f"\n{collection.collection_name} ({collection.collection_id}) — "
            f"{run.num_simulations:,} simulations",
            "",
            display.to_string(index=False),
            "",
            f"  Mean duplicates          {summary['mean_duplicates']:.1f} boxes "
            f"({summary['mean_duplicate_rate']:.1%} of purchases)",
            f"  Half the set ({collection.num_figures // 2} figures)"
            f"      {summary['boxes_for_half_the_set']:.1f} boxes",
            f"  Last figure alone        {summary['final_figure_mean_boxes']:.1f} boxes "
            f"= {currency} {summary['final_figure_mean_cost']:,.2f} "
            f"({summary['final_figure_share_of_boxes']:.1%} of the whole hunt)",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="all")
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--curve", action="store_true", help="print the unique-figures curve instead"
    )
    args = parser.parse_args()

    if args.collection.lower() == "all":
        collections = list(load_collections().values())
    else:
        collections = [load_collection(args.collection)]

    for collection in collections:
        run = simulate(collection, args.simulations, seed=args.seed)
        if args.curve:
            curve = unique_curve(collection, run=run, num_points=25)
            print(f"\n{collection.collection_name} ({collection.collection_id})")
            print(curve.round(3).to_string(index=False))
        else:
            print(_format(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
