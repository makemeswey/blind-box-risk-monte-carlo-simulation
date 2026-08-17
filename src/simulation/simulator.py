from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.database.connection import apply_schema, get_connection
from src.probability import coupon_collector as cc
from src.probability.distributions import Collection, load_collection, load_collections

BATCH_SIMULATIONS = 25_000

RESULT_COLUMNS = ["run_id", "simulation_id", "boxes_required", "duplicates", "total_cost"]

@dataclass(frozen=True)
class SimulationRun:
    collection: Collection
    results: pd.DataFrame
    num_simulations: int
    seed: int | None
    budget: float | None = None
    params: dict | None = None
    # (num_simulations, num_figures): box index at which the k-th distinct
    # figure arrived. The last column is boxes_required. Feeds the duplicate
    # analysis; set track_milestones=False to drop it on very large runs.
    milestones: np.ndarray | None = None

    @property
    def boxes(self) -> np.ndarray:
        return self.results["boxes_required"].to_numpy()

    def require_milestones(self) -> np.ndarray:
        if self.milestones is None:
            raise ValueError("Run was simulated with track_milestones=False")
        return self.milestones

    @property
    def cost(self) -> np.ndarray:
        return self.results["total_cost"].to_numpy()

    def completion_probability(self, budget: float | None = None) -> float:
        """Share of simulations completed for `budget` or less."""
        budget = self.budget if budget is None else budget
        if budget is None:
            raise ValueError("No budget given")
        return float((self.cost <= budget).mean())


def _chunk_size(collection: Collection) -> int:
    expected = cc.expected_boxes(collection.probabilities)
    return int(max(2 * collection.num_figures, round(1.5 * expected)))


def _simulate_batch(collection: Collection, num_simulations: int, rng: np.random.Generator, chunk: int) -> np.ndarray:
    """Milestones for `num_simulations` runs: box index of each k-th new figure."""
    n = collection.num_figures
    first_seen = np.full((num_simulations, n), -1, dtype=np.int64)
    active = np.arange(num_simulations)
    opened = 0

    while active.size:
        draws = collection.sample(rng, (active.size, chunk))

        local = np.full((active.size, n), -1, dtype=np.int64)
        rows = np.arange(active.size)
        for box in range(chunk - 1, -1, -1):
            local[rows, draws[:, box]] = box

        seen = first_seen[active]
        newly_seen = (seen < 0) & (local >= 0)
        seen[newly_seen] = local[newly_seen] + opened
        first_seen[active] = seen

        opened += chunk
        active = active[(first_seen[active] < 0).any(axis=1)]

    # Sorting each row turns "which box first showed figure i" into "which box
    # produced the k-th distinct figure" — the collection trajectory.
    return np.sort(first_seen, axis=1) + 1


def simulate(collection: Collection, num_simulations: int = 10_000, seed: int | None = None, budget: float | None = None, params: dict | None = None, batch_simulations: int = BATCH_SIMULATIONS, track_milestones: bool = True) -> SimulationRun:
    if num_simulations < 1:
        raise ValueError("num_simulations must be >= 1")

    rng = np.random.default_rng(seed)
    chunk = _chunk_size(collection)

    batches = []
    remaining = num_simulations
    while remaining > 0:
        size = min(batch_simulations, remaining)
        batches.append(_simulate_batch(collection, size, rng, chunk))
        remaining -= size

    milestones = np.concatenate(batches)
    boxes = milestones[:, -1]
    n = collection.num_figures
    results = pd.DataFrame(
        {
            "simulation_id": np.arange(1, num_simulations + 1),
            "boxes_required": boxes,
            "unique_figures": n,
            "duplicates": boxes - n,
            "duplicate_rate": (boxes - n) / boxes,
            "total_cost": boxes * collection.box_price,
        }
    )
    return SimulationRun(
        collection,
        results,
        num_simulations,
        seed,
        budget,
        params,
        milestones if track_milestones else None,
    )


def save_run(conn, run: SimulationRun) -> int:
    collection = run.collection
    try:
        conn.execute("BEGIN TRANSACTION")
        run_id = conn.execute(
            """
            INSERT INTO simulation_runs
                (collection_id, num_simulations, budget, box_price, seed, params)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING run_id
            """,
            [
                collection.collection_id,
                run.num_simulations,
                run.budget,
                collection.box_price,
                run.seed,
                json.dumps(run.params) if run.params else None,
            ],
        ).fetchone()[0]

        payload = run.results.assign(run_id=run_id)[RESULT_COLUMNS]
        conn.register("incoming_results", payload)
        conn.execute(
            f"INSERT INTO simulation_results ({', '.join(RESULT_COLUMNS)}) "
            f"SELECT {', '.join(RESULT_COLUMNS)} FROM incoming_results"
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister("incoming_results")

    return int(run_id)


def compare_with_analytical(run: SimulationRun) -> dict:
    collection = run.collection
    probabilities = collection.probabilities
    boxes = run.boxes

    analytical_mean = cc.expected_boxes(probabilities)
    simulated_mean = float(boxes.mean())
    standard_error = float(boxes.std(ddof=1) / np.sqrt(run.num_simulations))

    comparison = {
        "collection": collection.collection_name,
        "num_simulations": run.num_simulations,
        "analytical_expected_boxes": analytical_mean,
        "simulated_expected_boxes": simulated_mean,
        "standard_error": standard_error,
        "z_score": (simulated_mean - analytical_mean) / standard_error
        if standard_error
        else float("nan"),
        "analytical_median_boxes": cc.boxes_for_quantile(probabilities, 0.50),
        "simulated_median_boxes": float(np.median(boxes)),
        "analytical_p95_boxes": cc.boxes_for_quantile(probabilities, 0.95),
        "simulated_p95_boxes": float(np.percentile(boxes, 95)),
        "uniform_expected_boxes": cc.expected_boxes_uniform(collection.num_figures),
    }

    if run.budget is not None:
        affordable = int(run.budget // collection.box_price)
        comparison["budget"] = run.budget
        comparison["analytical_completion_probability"] = cc.completion_probability(
            probabilities, affordable
        )
        comparison["simulated_completion_probability"] = run.completion_probability()
    return comparison


def _report(run: SimulationRun) -> str:
    collection = run.collection
    boxes, cost = run.boxes, run.cost
    comparison = compare_with_analytical(run)
    currency = collection.currency

    lines = [
        f"\n{collection.collection_name} ({collection.collection_id}) — "
        f"{run.num_simulations:,} simulations, seed={run.seed}",
        f"  {collection.num_figures} figures, {currency} {collection.box_price:.2f} per box",
        "",
        f"  Expected boxes      {boxes.mean():10.2f}   "
        f"(analytical {comparison['analytical_expected_boxes']:.2f}, "
        f"z = {comparison['z_score']:+.2f})",
        f"  Median boxes        {np.median(boxes):10.2f}   "
        f"(analytical {comparison['analytical_median_boxes']})",
        f"  P95 boxes           {np.percentile(boxes, 95):10.2f}   "
        f"(analytical {comparison['analytical_p95_boxes']})",
        f"  Duplicate rate      {run.results['duplicate_rate'].mean():10.2%}",
        "",
        f"  Expected cost       {currency} {cost.mean():10.2f}",
        f"  Median cost         {currency} {np.median(cost):10.2f}",
        f"  P95 cost            {currency} {np.percentile(cost, 95):10.2f}",
    ]

    if run.budget is not None:
        lines += [
            "",
            f"  P(complete | budget {currency} {run.budget:,.2f})"
            f"  {run.completion_probability():.2%}   "
            f"(analytical {comparison['analytical_completion_probability']:.2%})",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default="all",
        help="collection_id to simulate, or 'all' (default)",
    )
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save", action="store_true", help="persist the run into DuckDB"
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
            run = simulate(
                collection,
                num_simulations=args.simulations,
                seed=args.seed,
                budget=args.budget,
            )
            print(_report(run))
            if args.save:
                print(f"\n  Saved as run_id {save_run(conn, run)}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
