"""Sensitivity analysis (section 16): what actually drives completion risk.

Five levers are swept, each by building a variant collection and re-evaluating
it: box price, secret probability, collection size, number of rare figures, and
how unequal the regular figures are. Every variant is renormalised, so each one
is a valid distribution rather than a fudged version of the original.

Because the coupon collector results are exact for unequal probabilities, the
sweeps are analytical by default — instant, noise-free, and safe to drive from
a dashboard slider. Pass `num_simulations` to add Monte Carlo columns.

    python -m src.analysis.sensitivity --collection JJK --budget 5000
    python -m src.analysis.sensitivity --collection JJK --parameter secret_probability
    python -m src.analysis.sensitivity --collection JJK --budget 5000 --question
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.probability import coupon_collector as cc
from src.probability.distributions import Collection, load_collection, load_collections
from src.simulation import metrics
from src.simulation.experiments import with_secret_probability
from src.simulation.simulator import simulate

# Default probability given to each figure promoted to "rare": twice as hard to
# find as a regular in a 12-figure set.
DEFAULT_RARE_PROBABILITY = 1 / 24

DEFAULT_SEED = 42


# --------------------------------------------------------------------------
# Variant builders
# --------------------------------------------------------------------------
def _regular_indices(collection: Collection) -> list[int]:
    secrets = set(collection.secret_indices)
    return [i for i in range(collection.num_figures) if i not in secrets]


def resize_regulars(collection: Collection, num_regulars: int) -> Collection:
    """Change how many regular figures the collection has, keeping the secret.

    Existing figure names are reused where possible and placeholders added
    beyond them, since the variant is hypothetical.
    """
    if num_regulars < 1:
        raise ValueError("A collection needs at least one regular figure")

    secrets = collection.secret_indices
    secret_mass = float(collection.probabilities[list(secrets)].sum()) if secrets else 0.0
    regular_share = (1 - secret_mass) / num_regulars

    ids, names, rarities, probabilities = [], [], [], []
    existing = _regular_indices(collection)
    for position in range(num_regulars):
        if position < len(existing):
            source = existing[position]
            ids.append(collection.figure_ids[source])
            names.append(collection.figure_names[source])
        else:
            ids.append(f"{collection.collection_id}X{position + 1:02d}")
            names.append(f"Regular {position + 1}")
        rarities.append("regular")
        probabilities.append(regular_share)

    for index in secrets:
        ids.append(collection.figure_ids[index])
        names.append(collection.figure_names[index])
        rarities.append("secret")
        probabilities.append(float(collection.probabilities[index]))

    return collection.replace(
        figure_ids=tuple(ids),
        figure_names=tuple(names),
        rarities=tuple(rarities),
        probabilities=np.array(probabilities),
    )


def with_rare_figures(
    collection: Collection,
    num_rare: int,
    rare_probability: float = DEFAULT_RARE_PROBABILITY,
) -> Collection:
    """Promote `num_rare` regular figures to "rare" at `rare_probability` each."""
    regulars = _regular_indices(collection)
    if not 0 <= num_rare <= len(regulars):
        raise ValueError(f"num_rare must be between 0 and {len(regulars)}")

    probabilities = collection.probabilities.copy()
    rarities = list(collection.rarities)

    regular_mass = float(probabilities[regulars].sum())
    rare_mass = num_rare * rare_probability
    if rare_mass >= regular_mass:
        raise ValueError(
            f"{num_rare} rare figures at {rare_probability:.4f} need {rare_mass:.4f}, "
            f"but only {regular_mass:.4f} is available"
        )

    promoted = regulars[:num_rare]
    remaining = regulars[num_rare:]
    if not remaining and num_rare:
        raise ValueError("At least one figure must stay regular")

    probabilities[promoted] = rare_probability
    probabilities[remaining] *= (regular_mass - rare_mass) / probabilities[
        remaining
    ].sum()
    for index in promoted:
        rarities[index] = "rare"

    return collection.replace(probabilities=probabilities, rarities=tuple(rarities))


def with_skew(collection: Collection, exponent: float) -> Collection:
    """Make the regular figures unequally likely: weight_i proportional to i^-exponent.

    exponent = 0 is the uniform case; larger values concentrate probability on
    the first few figures and starve the rest, which is what drives completion
    cost up even without a secret.
    """
    regulars = _regular_indices(collection)
    probabilities = collection.probabilities.copy()

    regular_mass = float(probabilities[regulars].sum())
    weights = np.arange(1, len(regulars) + 1, dtype=float) ** -exponent
    probabilities[regulars] = regular_mass * weights / weights.sum()
    return collection.replace(probabilities=probabilities)


PARAMETERS = {
    "box_price": lambda c, v, **_: c.with_box_price(v),
    "secret_probability": lambda c, v, **_: with_secret_probability(c, v),
    # Sweep value is the total figure count; the secrets are held fixed, so the
    # regulars absorb the change.
    "collection_size": lambda c, v, **_: resize_regulars(
        c, int(v) - len(c.secret_indices)
    ),
    "num_rare": lambda c, v, **options: with_rare_figures(
        c, int(v), options.get("rare_probability", DEFAULT_RARE_PROBABILITY)
    ),
    "skew": lambda c, v, **_: with_skew(c, v),
}


def default_values(collection: Collection, parameter: str) -> np.ndarray:
    """A sensible sweep range for each parameter."""
    if parameter == "box_price":
        return np.round(collection.box_price * np.array([0.5, 0.75, 1.0, 1.5, 2.0]), 2)
    if parameter == "secret_probability":
        return np.array([1 / d for d in (72, 100, 144, 200, 288, 500)])
    if parameter == "collection_size":
        return np.array([6, 8, 10, 12, 16, 20])
    if parameter == "num_rare":
        return np.arange(0, 5)
    if parameter == "skew":
        return np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    raise ValueError(f"Unknown parameter {parameter!r}; choose from {list(PARAMETERS)}")


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------
def _evaluate(
    variant: Collection,
    budget: float | None,
    num_simulations: int,
    seed: int | None,
) -> dict:
    summary = cc.analytical_summary(variant, budget)
    row = {
        "num_figures": variant.num_figures,
        "box_price": variant.box_price,
        "expected_boxes": summary["expected_boxes"],
        "median_boxes": summary["median_boxes"],
        "p95_boxes": summary["p95_boxes"],
        "expected_cost": summary["expected_cost"],
        "p95_cost": summary["p95_cost"],
        "duplicate_rate": 1 - variant.num_figures / summary["expected_boxes"],
        "completion_probability": summary.get("completion_probability", float("nan")),
    }

    if num_simulations:
        run = simulate(
            variant, num_simulations, seed=seed, budget=budget, track_milestones=False
        )
        simulated = metrics.summarise(run, budget)
        row.update(
            {
                "sim_expected_boxes": simulated["mean_boxes"],
                "sim_expected_cost": simulated["mean_cost"],
                "sim_p95_cost": simulated["p95_cost"],
                "sim_completion_probability": simulated.get(
                    "completion_probability", float("nan")
                ),
                "standard_error_boxes": simulated["standard_error_boxes"],
            }
        )
    return row


def sweep(
    collection: Collection,
    parameter: str,
    values=None,
    budget: float | None = None,
    num_simulations: int = 0,
    seed: int | None = DEFAULT_SEED,
    **options,
) -> pd.DataFrame:
    """Vary one parameter and report how the risk profile responds.

    Results are indexed by the parameter value and include `cost_vs_baseline`,
    the expected cost relative to the collection as it is actually sold.
    """
    if parameter not in PARAMETERS:
        raise ValueError(f"Unknown parameter {parameter!r}; choose from {list(PARAMETERS)}")
    if values is None:
        values = default_values(collection, parameter)

    build = PARAMETERS[parameter]
    rows = []
    for value in values:
        variant = build(collection, value, **options)
        rows.append({"parameter": parameter, "value": value, **_evaluate(
            variant, budget, num_simulations, seed
        )})

    frame = pd.DataFrame(rows)
    baseline = cc.analytical_summary(collection)["expected_cost"]
    frame["cost_vs_baseline"] = frame["expected_cost"] / baseline
    return frame


def sweep_all(
    collection: Collection,
    budget: float | None = None,
    num_simulations: int = 0,
    seed: int | None = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    return {
        parameter: sweep(collection, parameter, budget=budget,
                         num_simulations=num_simulations, seed=seed)
        for parameter in PARAMETERS
    }


def elasticity(frame: pd.DataFrame, metric: str = "expected_cost") -> float:
    """Rough % change in `metric` per % change in the swept parameter.

    A log-log slope across the sweep range: ~1 means proportional, >1 means the
    parameter is amplified. Only meaningful for positive-valued parameters.
    """
    values = frame["value"].to_numpy(dtype=float)
    outputs = frame[metric].to_numpy(dtype=float)
    usable = (values > 0) & (outputs > 0)
    if usable.sum() < 2:
        return float("nan")
    slope = np.polyfit(np.log(values[usable]), np.log(outputs[usable]), 1)[0]
    return float(slope)


def secret_change_impact(
    collection: Collection,
    budget: float,
    from_denominator: int = 144,
    to_denominator: int = 288,
) -> dict:
    """Answer the worked question in section 16 directly.

    "How much does reducing the secret probability from 1/144 to 1/288 affect
    the probability of completing the collection within a given budget?"
    """
    before = with_secret_probability(collection, 1 / from_denominator)
    after = with_secret_probability(collection, 1 / to_denominator)

    before_summary = cc.analytical_summary(before, budget)
    after_summary = cc.analytical_summary(after, budget)

    return {
        "collection_id": collection.collection_id,
        "budget": budget,
        "boxes_affordable": before_summary["boxes_affordable"],
        "from": f"1/{from_denominator}",
        "to": f"1/{to_denominator}",
        "completion_before": before_summary["completion_probability"],
        "completion_after": after_summary["completion_probability"],
        "completion_change": after_summary["completion_probability"]
        - before_summary["completion_probability"],
        "expected_boxes_before": before_summary["expected_boxes"],
        "expected_boxes_after": after_summary["expected_boxes"],
        "expected_cost_before": before_summary["expected_cost"],
        "expected_cost_after": after_summary["expected_cost"],
        "cost_multiplier": after_summary["expected_cost"]
        / before_summary["expected_cost"],
    }


DISPLAY_COLUMNS = [
    "value",
    "num_figures",
    "box_price",
    "expected_boxes",
    "median_boxes",
    "p95_boxes",
    "expected_cost",
    "p95_cost",
    "duplicate_rate",
    "completion_probability",
    "cost_vs_baseline",
    # Present only when --simulations is used.
    "sim_expected_boxes",
    "sim_expected_cost",
    "sim_completion_probability",
]


def _format_sweep(frame: pd.DataFrame, collection: Collection) -> str:
    parameter = frame["parameter"].iloc[0]
    display = frame[[c for c in DISPLAY_COLUMNS if c in frame]].copy()
    if parameter == "secret_probability":
        display.insert(1, "odds_1_in", (1 / display["value"]).round(0))
    display = display.round(
        {
            "value": 5,
            "expected_boxes": 1,
            "p95_boxes": 0,
            "expected_cost": 2,
            "p95_cost": 2,
            "duplicate_rate": 4,
            "completion_probability": 4,
            "cost_vs_baseline": 2,
            "sim_expected_boxes": 1,
            "sim_expected_cost": 2,
            "sim_completion_probability": 4,
        }
    )
    slope = elasticity(frame)
    return "\n".join(
        [
            f"\n  {parameter}  ({collection.currency}, elasticity of expected cost "
            f"= {slope:+.2f})",
            display.to_string(index=False),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="all")
    parser.add_argument(
        "--parameter",
        default="all",
        help=f"one of {list(PARAMETERS)}, or 'all' (default)",
    )
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument(
        "--simulations",
        type=int,
        default=0,
        help="add Monte Carlo columns (0 = analytical only)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--question",
        action="store_true",
        help="answer the 1/144 -> 1/288 budget question for --budget",
    )
    args = parser.parse_args()

    if args.collection.lower() == "all":
        collections = list(load_collections().values())
    else:
        collections = [load_collection(args.collection)]

    for collection in collections:
        print(
            f"\n{collection.collection_name} ({collection.collection_id}) — "
            f"baseline {collection.num_figures} figures at "
            f"{collection.currency} {collection.box_price:.2f}"
        )

        if args.question:
            if args.budget is None:
                parser.error("--question needs --budget")
            impact = secret_change_impact(collection, args.budget)
            print(
                f"\n  Secret {impact['from']} -> {impact['to']} at "
                f"{collection.currency} {impact['budget']:,.2f} "
                f"({impact['boxes_affordable']} boxes):\n"
                f"    P(complete)   {impact['completion_before']:.2%} -> "
                f"{impact['completion_after']:.2%} "
                f"({impact['completion_change']:+.2%})\n"
                f"    Expected cost {collection.currency} "
                f"{impact['expected_cost_before']:,.2f} -> "
                f"{collection.currency} {impact['expected_cost_after']:,.2f} "
                f"({impact['cost_multiplier']:.2f}x)"
            )
            continue

        parameters = (
            list(PARAMETERS) if args.parameter.lower() == "all" else [args.parameter]
        )
        for parameter in parameters:
            frame = sweep(
                collection,
                parameter,
                budget=args.budget,
                num_simulations=args.simulations,
                seed=args.seed,
            )
            print(_format_sweep(frame, collection))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
