"""Page 2 — run a Monte Carlo simulation and read the headline numbers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis import duplicates
from src.dashboard import data, theme
from src.probability.distributions import Collection
from src.simulation import metrics
from src.simulation.simulator import SimulationRun, compare_with_analytical

SESSION_KEY = "simulation_params"


def params_for(collection: Collection) -> dict:
    """The run settings currently in play — page 3 reads the same ones."""
    stored = st.session_state.get(SESSION_KEY)
    if stored and stored["collection_id"] == collection.collection_id:
        return stored
    return {
        "collection_id": collection.collection_id,
        "num_simulations": data.DEFAULT_SIMULATIONS,
        "budget": data.default_budget(collection),
        "seed": data.DEFAULT_SEED,
    }


def controls(collection: Collection) -> dict:
    """Simulation settings; the form only submits when the button is pressed."""
    current = params_for(collection)

    with st.form("simulation_controls"):
        left, middle, right = st.columns(3)
        num_simulations = left.slider(
            "Number of simulations",
            min_value=data.MIN_SIMULATIONS,
            max_value=data.MAX_SIMULATIONS,
            value=data.clamp_simulations(current["num_simulations"]),
            step=data.SIMULATION_STEP,
            help="Any count in this range; the mean is already stable well before the top.",
        )
        budget = middle.number_input(
            f"Budget ({theme.symbol(collection.currency).strip()})",
            min_value=float(collection.box_price),
            value=float(current["budget"]),
            step=100.0,
            help="Completion probability is the share of simulations finishing at or below this.",
        )
        seed = right.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000,
            value=int(current["seed"]),
            help="Same seed, same draws — the run is reproducible.",
        )
        submitted = st.form_submit_button("▶  Run Simulation", type="primary")

    params = {
        "collection_id": collection.collection_id,
        "num_simulations": int(num_simulations),
        "budget": float(budget),
        "seed": int(seed),
    }
    if submitted:
        st.session_state[SESSION_KEY] = params
    return params


def headline_metrics(summary: dict, currency: str) -> None:
    tail_premium = summary["p95_cost"] / summary["mean_cost"] - 1

    boxes, median, cost, tail = st.columns(4)
    boxes.metric("Expected boxes", f"{summary['mean_boxes']:.0f}")
    median.metric("Median boxes", f"{summary['median_boxes']:.0f}")
    cost.metric("Expected cost", theme.money(summary["mean_cost"], currency, 0))
    tail.metric(
        "P95 cost",
        theme.money(summary["p95_cost"], currency, 0),
        delta=f"{tail_premium:+.0%} tail risk",
        delta_color="inverse",
    )


def completion_metrics(summary: dict, currency: str) -> None:
    completion, dupes, rate = st.columns(3)
    completion.metric(
        f"Completion P(≤ {theme.money(summary['budget'], currency, 0)})",
        f"{summary['completion_probability']:.1%}",
        delta=f"{summary['boxes_affordable']} boxes affordable",
        delta_color="off",
    )
    dupes.metric("Avg duplicates", f"{summary['mean_duplicates']:.0f}")
    rate.metric("Duplicate rate", f"{summary['mean_duplicate_rate']:.1%}")


def duplicate_note(run: SimulationRun) -> None:
    summary = duplicates.duplicate_summary(run)
    currency = run.collection.currency
    st.caption(
        f"The **last** missing figure alone takes "
        f"{summary['final_figure_mean_boxes']:.0f} boxes on average "
        f"({theme.money(summary['final_figure_mean_cost'], currency, 0)}, "
        f"{summary['final_figure_share_of_boxes']:.0%} of the whole run) — "
        f"half the set arrives in the first {summary['boxes_for_half_the_set']:.0f} boxes."
    )


def unique_figures_chart(run: SimulationRun) -> None:
    st.subheader("Unique figures vs. boxes opened")
    curve = duplicates.unique_curve(run.collection, run=run).rename(
        columns={
            "expected_unique": "Analytical E[unique]",
            "simulated_unique": "Simulated mean",
        }
    )
    st.line_chart(
        curve,
        x="boxes",
        y=["Simulated mean", "Analytical E[unique]"],
        x_label="Boxes opened",
        y_label="Distinct figures owned",
        color=[theme.ACCENT, theme.BLUE],
        height=280,
    )
    st.caption(
        "The two curves sit on top of each other — the simulation reproduces the exact "
        "expectation. Both flatten because every figure already owned makes the next "
        "box less likely to contain something new."
    )


def validation_table(run: SimulationRun) -> None:
    st.subheader("Validation against the coupon collector model")
    check = compare_with_analytical(run)
    frame = pd.DataFrame(
        [
            {
                "Quantity": "Expected boxes",
                "Analytical": check["analytical_expected_boxes"],
                "Simulated": check["simulated_expected_boxes"],
            },
            {
                "Quantity": "Median boxes",
                "Analytical": check["analytical_median_boxes"],
                "Simulated": check["simulated_median_boxes"],
            },
            {
                "Quantity": "P95 boxes",
                "Analytical": check["analytical_p95_boxes"],
                "Simulated": check["simulated_p95_boxes"],
            },
        ]
    )
    frame["Difference"] = frame["Simulated"] - frame["Analytical"]

    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            "Analytical": st.column_config.NumberColumn(format="%.2f"),
            "Simulated": st.column_config.NumberColumn(format="%.2f"),
            "Difference": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

    z = check["z_score"]
    message = (
        f"Simulated mean is {abs(z):.2f} standard errors from the exact value "
        f"(±{check['standard_error']:.2f} boxes at {run.num_simulations:,} simulations)."
    )
    if abs(z) <= 2:
        st.success(message)
    else:
        st.warning(message + " Worth re-running with a different seed.")


def render(collection: Collection) -> None:
    theme.page_header(
        "Run Simulation",
        f"Configure and run a Monte Carlo simulation of completing "
        f"{collection.collection_name} at "
        f"{theme.money(collection.box_price, collection.currency)} per box.",
    )

    params = controls(collection)
    run = data.run_simulation(
        params["collection_id"],
        params["num_simulations"],
        params["budget"],
        params["seed"],
    )
    summary = metrics.summarise(run, params["budget"])

    headline_metrics(summary, collection.currency)
    completion_metrics(summary, collection.currency)
    duplicate_note(run)

    unique_figures_chart(run)
    validation_table(run)
