from __future__ import annotations
import argparse
from statistics import NormalDist
import numpy as np
import pandas as pd
from src.database.connection import get_connection
from src.simulation.simulator import SimulationRun

PERCENTILES = (75, 90, 95, 99)
DEFAULT_CONFIDENCE = 0.95

def simulation_record(run: SimulationRun, simulation_id: int) -> dict:
    """The full record for a single simulation."""
    matches = run.results[run.results["simulation_id"] == simulation_id]
    if matches.empty:
        raise ValueError(f"No simulation {simulation_id} in this run")

    row = matches.iloc[0]
    return {
        "collection_id": run.collection.collection_id,
        "collection_name": run.collection.collection_name,
        "simulation_id": int(row["simulation_id"]),
        "boxes_required": int(row["boxes_required"]),
        "unique_figures": int(row["unique_figures"]),
        "duplicates": int(row["duplicates"]),
        "duplicate_rate": float(row["duplicate_rate"]),
        "total_cost": float(row["total_cost"]),
    }


def format_record(record: dict, currency: str = "MYR") -> str:
    return "\n".join(
        [
            f"Simulation {record['simulation_id']}  ({record['collection_name']})",
            f"  Boxes:       {record['boxes_required']}",
            f"  Unique:      {record['unique_figures']}",
            f"  Duplicates:  {record['duplicates']} ({record['duplicate_rate']:.1%})",
            f"  Cost:        {currency} {record['total_cost']:,.2f}",
        ]
    )


# --------------------------------------------------------------------------
# Section 13 — aggregate statistics
# --------------------------------------------------------------------------
def _distribution_stats(values: np.ndarray, prefix: str) -> dict:
    """Central tendency, dispersion and tail percentiles for one quantity."""
    values = np.asarray(values, dtype=float)
    stats = {
        f"mean_{prefix}": float(values.mean()),
        f"median_{prefix}": float(np.median(values)),
        f"std_{prefix}": float(values.std(ddof=1)),
        f"var_{prefix}": float(values.var(ddof=1)),
        f"min_{prefix}": float(values.min()),
        f"max_{prefix}": float(values.max()),
    }
    for percentile in PERCENTILES:
        stats[f"p{percentile}_{prefix}"] = float(np.percentile(values, percentile))
    return stats


def mean_confidence_interval(
    values, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float, float]:
    """(standard error, low, high) for the mean.

    Monte Carlo error, not spread of the distribution: it says how precisely
    this many simulations pin down the expected value.
    """
    values = np.asarray(values, dtype=float)
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    mean = float(values.mean())
    return standard_error, mean - z * standard_error, mean + z * standard_error


def completion_probability(cost, budget: float) -> float:
    """P(total cost <= budget)."""
    return float((np.asarray(cost, dtype=float) <= budget).mean())


def completion_interval(
    cost, budget: float, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    cost = np.asarray(cost, dtype=float)
    n = cost.size
    p = completion_probability(cost, budget)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)

    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return float(max(0.0, centre - spread)), float(min(1.0, centre + spread))


def budget_for_confidence(cost, confidence: float = DEFAULT_CONFIDENCE) -> float:
    return float(np.percentile(np.asarray(cost, dtype=float), 100 * confidence))


def completion_curve(cost, budgets=None, num_points: int = 200) -> pd.DataFrame:
    cost = np.sort(np.asarray(cost, dtype=float))
    if budgets is None:
        budgets = np.linspace(cost[0], np.percentile(cost, 99), num_points)
    budgets = np.asarray(budgets, dtype=float)

    probabilities = np.searchsorted(cost, budgets, side="right") / cost.size
    return pd.DataFrame({"budget": budgets, "completion_probability": probabilities})


def summarise(
    run: SimulationRun,
    budget: float | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Full statistical summary of one simulation run."""
    boxes, cost = run.boxes, run.cost
    results = run.results
    collection = run.collection

    standard_error, ci_low, ci_high = mean_confidence_interval(boxes, confidence)

    summary = {
        "collection_id": collection.collection_id,
        "collection_name": collection.collection_name,
        "num_figures": collection.num_figures,
        "box_price": collection.box_price,
        "currency": collection.currency,
        "num_simulations": run.num_simulations,
        "seed": run.seed,
        **_distribution_stats(boxes, "boxes"),
        **_distribution_stats(cost, "cost"),
        "mean_duplicates": float(results["duplicates"].mean()),
        "median_duplicates": float(results["duplicates"].median()),
        "mean_duplicate_rate": float(results["duplicate_rate"].mean()),
        "pooled_duplicate_rate": float(
            results["duplicates"].sum() / results["boxes_required"].sum()
        ),
        "standard_error_boxes": standard_error,
        "ci_low_mean_boxes": ci_low,
        "ci_high_mean_boxes": ci_high,
        f"budget_for_{int(confidence * 100)}pct": budget_for_confidence(
            cost, confidence
        ),
    }

    budget = run.budget if budget is None else budget
    if budget is not None:
        low, high = completion_interval(cost, budget, confidence)
        summary["budget"] = float(budget)
        summary["boxes_affordable"] = int(budget // collection.box_price)
        summary["completion_probability"] = completion_probability(cost, budget)
        summary["completion_ci_low"] = low
        summary["completion_ci_high"] = high
    return summary


def summary_frame(runs, budget: float | None = None) -> pd.DataFrame:
    return pd.DataFrame([summarise(run, budget) for run in runs])


def format_summary(summary: dict) -> str:
    currency = summary.get("currency", "")
    lines = [
        f"\n{summary['collection_name']} ({summary['collection_id']}) — "
        f"{summary['num_simulations']:,} simulations",
        "",
        "  Boxes required",
        f"    Mean        {summary['mean_boxes']:10.2f}  "
        f"(95% CI {summary['ci_low_mean_boxes']:.2f} – {summary['ci_high_mean_boxes']:.2f})",
        f"    Median      {summary['median_boxes']:10.2f}",
        f"    Std dev     {summary['std_boxes']:10.2f}",
        f"    Variance    {summary['var_boxes']:10.2f}",
        f"    P75/P90     {summary['p75_boxes']:10.0f} / {summary['p90_boxes']:.0f}",
        f"    P95/P99     {summary['p95_boxes']:10.0f} / {summary['p99_boxes']:.0f}",
        f"    Worst case  {summary['max_boxes']:10.0f}",
        "",
        "  Expenditure",
        f"    Expected    {currency} {summary['mean_cost']:12,.2f}",
        f"    Median      {currency} {summary['median_cost']:12,.2f}",
        f"    P95         {currency} {summary['p95_cost']:12,.2f}",
        f"    P99         {currency} {summary['p99_cost']:12,.2f}",
        "",
        "  Duplicates",
        f"    Mean        {summary['mean_duplicates']:10.2f} boxes wasted",
        f"    Rate        {summary['mean_duplicate_rate']:10.2%} "
        f"(pooled {summary['pooled_duplicate_rate']:.2%})",
    ]

    if "completion_probability" in summary:
        lines += [
            "",
            f"  P(complete | budget {currency} {summary['budget']:,.2f})"
            f"   {summary['completion_probability']:.2%}  "
            f"(95% CI {summary['completion_ci_low']:.2%} – "
            f"{summary['completion_ci_high']:.2%})",
        ]

    key = next(k for k in summary if k.startswith("budget_for_"))
    lines.append(
        f"  Budget for {key.removeprefix('budget_for_').removesuffix('pct')}% "
        f"confidence   {currency} {summary[key]:,.2f}"
    )
    return "\n".join(lines)

def _percentile_sql(column: str, prefix: str) -> str:
    return ",\n        ".join(
        f"quantile_cont({column}, {p / 100}) AS p{p}_{prefix}" for p in PERCENTILES
    )


def summarise_sql(conn, run_id: int, budget: float | None = None) -> dict:
    run = conn.execute(
        "SELECT collection_id, num_simulations, budget, box_price, seed "
        "FROM simulation_runs WHERE run_id = ?",
        [run_id],
    ).fetchone()
    if run is None:
        raise ValueError(f"No simulation run with run_id {run_id}")

    collection_id, num_simulations, stored_budget, box_price, seed = run
    budget = stored_budget if budget is None else budget

    summary = conn.execute(
        f"""
        SELECT
            count(*)                                   AS num_simulations,
            avg(boxes_required)                        AS mean_boxes,
            quantile_cont(boxes_required, 0.5)         AS median_boxes,
            stddev_samp(boxes_required)                AS std_boxes,
            var_samp(boxes_required)                   AS var_boxes,
            min(boxes_required)                        AS min_boxes,
            max(boxes_required)                        AS max_boxes,
            {_percentile_sql("boxes_required", "boxes")},
            avg(total_cost)                            AS mean_cost,
            quantile_cont(total_cost, 0.5)             AS median_cost,
            stddev_samp(total_cost)                    AS std_cost,
            var_samp(total_cost)                       AS var_cost,
            min(total_cost)                            AS min_cost,
            max(total_cost)                            AS max_cost,
            {_percentile_sql("total_cost", "cost")},
            avg(duplicates)                            AS mean_duplicates,
            quantile_cont(duplicates, 0.5)             AS median_duplicates,
            avg(duplicates / boxes_required)           AS mean_duplicate_rate,
            sum(duplicates) / sum(boxes_required)      AS pooled_duplicate_rate,
            stddev_samp(boxes_required) / sqrt(count(*)) AS standard_error_boxes
        FROM simulation_results
        WHERE run_id = ?
        """,
        [run_id],
    ).df().iloc[0].to_dict()

    figures = conn.execute(
        "SELECT any_value(collection_name) AS collection_name, "
        "count(*) AS num_figures, any_value(currency) AS currency "
        "FROM figures WHERE collection_id = ?",
        [collection_id],
    ).fetchone()

    # .iloc[0] on a mixed-dtype row yields floats; restore the counts.
    for key in ("num_simulations", "min_boxes", "max_boxes"):
        summary[key] = int(summary[key])

    summary.update(
        {
            "run_id": run_id,
            "collection_id": collection_id,
            "collection_name": figures[0] if figures else collection_id,
            "num_figures": figures[1] if figures else None,
            "currency": figures[2] if figures else "",
            "box_price": box_price,
            "seed": seed,
        }
    )

    z = NormalDist().inv_cdf(0.5 + DEFAULT_CONFIDENCE / 2)
    summary["ci_low_mean_boxes"] = summary["mean_boxes"] - z * summary["standard_error_boxes"]
    summary["ci_high_mean_boxes"] = summary["mean_boxes"] + z * summary["standard_error_boxes"]
    summary[f"budget_for_{int(DEFAULT_CONFIDENCE * 100)}pct"] = conn.execute(
        "SELECT quantile_cont(total_cost, ?) FROM simulation_results WHERE run_id = ?",
        [DEFAULT_CONFIDENCE, run_id],
    ).fetchone()[0]

    if budget is not None:
        probability, n = conn.execute(
            "SELECT avg(CASE WHEN total_cost <= ? THEN 1.0 ELSE 0.0 END), count(*) "
            "FROM simulation_results WHERE run_id = ?",
            [budget, run_id],
        ).fetchone()
        denominator = 1 + z**2 / n
        centre = (probability + z**2 / (2 * n)) / denominator
        spread = (
            z * np.sqrt(probability * (1 - probability) / n + z**2 / (4 * n**2))
        ) / denominator
        summary["budget"] = float(budget)
        summary["boxes_affordable"] = int(budget // box_price)
        summary["completion_probability"] = float(probability)
        summary["completion_ci_low"] = float(max(0.0, centre - spread))
        summary["completion_ci_high"] = float(min(1.0, centre + spread))

    return summary


def list_runs(conn) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT r.run_id, r.collection_id, r.num_simulations, r.budget,
               r.box_price, r.seed, r.created_at,
               avg(s.boxes_required) AS mean_boxes,
               avg(s.total_cost)     AS mean_cost
        FROM simulation_runs r
        JOIN simulation_results s USING (run_id)
        GROUP BY ALL
        ORDER BY r.run_id
        """
    ).df()


def compare_runs(conn, budget: float | None = None) -> pd.DataFrame:
    """One row per stored run — the section 17 comparison table."""
    run_ids = [row[0] for row in conn.execute(
        "SELECT run_id FROM simulation_runs ORDER BY run_id"
    ).fetchall()]
    return pd.DataFrame([summarise_sql(conn, run_id, budget) for run_id in run_ids])


COMPARISON_COLUMNS = [
    "run_id",
    "collection_id",
    "num_figures",
    "box_price",
    "mean_boxes",
    "median_boxes",
    "mean_cost",
    "p95_cost",
    "mean_duplicate_rate",
    "completion_probability",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, help="summarise one stored run")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument(
        "--compare-runs", action="store_true", help="one row per stored run"
    )
    parser.add_argument(
        "--record", type=int, default=None, help="print a single simulation record"
    )
    args = parser.parse_args()

    conn = get_connection(read_only=True)
    try:
        runs = list_runs(conn)
        if runs.empty:
            print("No stored runs; run `python -m src.simulation.simulator --save` first")
            return 1

        if args.compare_runs or args.run_id is None:
            comparison = compare_runs(conn, args.budget)
            available = [c for c in COMPARISON_COLUMNS if c in comparison.columns]
            print(comparison[available].to_string(index=False))
            return 0

        summary = summarise_sql(conn, args.run_id, args.budget)
        print(format_summary(summary))

        if args.record is not None:
            row = conn.execute(
                "SELECT * FROM simulation_results WHERE run_id = ? AND simulation_id = ?",
                [args.run_id, args.record],
            ).df()
            if row.empty:
                print(f"\nNo simulation {args.record} in run {args.run_id}")
            else:
                row = row.iloc[0]
                print(
                    "\n"
                    + format_record(
                        {
                            "simulation_id": int(row["simulation_id"]),
                            "collection_name": summary["collection_name"],
                            "boxes_required": int(row["boxes_required"]),
                            "unique_figures": summary["num_figures"],
                            "duplicates": int(row["duplicates"]),
                            "duplicate_rate": row["duplicates"] / row["boxes_required"],
                            "total_cost": float(row["total_cost"]),
                        },
                        summary["currency"],
                    )
                )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
