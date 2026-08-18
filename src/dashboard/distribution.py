"""Page 3 — the shape of the result: histograms, percentiles, budget curve."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard import data, theme
from src.dashboard.simulation import params_for
from src.probability.distributions import Collection
from src.simulation import metrics
from src.simulation.simulator import SimulationRun

PERCENTILES = (("P50", 50, theme.GREEN), ("P90", 90, theme.ORANGE), ("P95", 95, theme.ACCENT))


def run_caption(run: SimulationRun, budget: float) -> None:
    st.caption(
        f"{run.num_simulations:,} simulations of {run.collection.collection_name}, "
        f"seed {run.seed}, budget "
        f"{theme.money(budget, run.collection.currency, 0)} — change them on the "
        "Simulation page."
    )


CLIP_PERCENTILE = 99.0


def histogram(values: np.ndarray, x_label: str, colour: str, value_format: str) -> alt.Chart:
    """Histogram with dashed percentile rules, clipped at P99 so the bulk is readable."""
    limit = float(np.percentile(values, CLIP_PERCENTILE))
    frame = pd.DataFrame({"value": values[values <= limit]})
    bars = (
        alt.Chart(frame)
        .mark_bar(color=colour, opacity=0.9)
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=60), title=x_label),
            y=alt.Y("count()", title="Simulations"),
            tooltip=[
                alt.Tooltip("value:Q", bin=alt.Bin(maxbins=60), title=x_label, format=value_format),
                alt.Tooltip("count()", title="Simulations", format=","),
            ],
        )
    )

    markers = pd.DataFrame(
        [
            {"label": label, "value": float(np.percentile(values, percentile))}
            for label, percentile, _ in PERCENTILES
        ]
    )
    rules = (
        alt.Chart(markers)
        .mark_rule(strokeDash=[5, 3], size=2)
        .encode(
            x=alt.X("value:Q"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=[label for label, _, _ in PERCENTILES],
                    range=[colour for _, _, colour in PERCENTILES],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["label:N", alt.Tooltip("value:Q", format=value_format)],
        )
    )
    return (bars + rules).properties(height=300)


def histograms(run: SimulationRun) -> None:
    boxes, cost = st.columns(2)
    with boxes:
        st.subheader("Boxes required")
        st.altair_chart(
            histogram(run.boxes, "Boxes opened", theme.BLUE, ",.0f"), width="stretch"
        )
    with cost:
        st.subheader(f"Total expenditure ({run.collection.currency})")
        st.altair_chart(
            histogram(run.cost, "Total cost", theme.ACCENT, ",.0f"), width="stretch"
        )
    st.caption(
        f"Both distributions are right-skewed: the mean sits above the median because "
        f"a small share of collectors chase the last figure far longer than the rest. "
        f"The axes stop at P{CLIP_PERCENTILE:.0f} — the worst run of these "
        f"{run.num_simulations:,} needed {run.boxes.max():,} boxes "
        f"({theme.money(run.cost.max(), run.collection.currency, 0)})."
    )


def percentile_table(run: SimulationRun, summary: dict) -> None:
    st.subheader("Percentile summary")
    labels = {"P50 (median)": 50, "P75": 75, "P90": 90, "P95": 95, "P99": 99}
    frame = pd.DataFrame(
        [
            {
                "Percentile": label,
                "Boxes": float(np.percentile(run.boxes, percentile)),
                "Cost": float(np.percentile(run.cost, percentile)),
                "Duplicates": float(
                    np.percentile(run.results["duplicates"].to_numpy(), percentile)
                ),
            }
            for label, percentile in labels.items()
        ]
    )
    frame = frame.round(0).rename(columns={"Cost": f"Cost ({run.collection.currency})"})
    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            "Boxes": st.column_config.NumberColumn(format="localized"),
            f"Cost ({run.collection.currency})": st.column_config.NumberColumn(
                format="localized"
            ),
            "Duplicates": st.column_config.NumberColumn(format="localized"),
        },
    )
    st.caption(
        f"Mean {summary['mean_boxes']:.1f} boxes vs median {summary['median_boxes']:.0f} — "
        f"the gap is the tail. One collector in twenty spends at least "
        f"{theme.money(summary['p95_cost'], run.collection.currency, 0)}."
    )


def completion_curve(run: SimulationRun, budget: float) -> None:
    st.subheader("Probability of completing within a budget")
    curve = metrics.completion_curve(run.cost).rename(
        columns={"budget": "Budget", "completion_probability": "P(complete)"}
    )
    st.line_chart(
        curve,
        x="Budget",
        y="P(complete)",
        x_label=f"Budget ({run.collection.currency})",
        y_label="P(complete)",
        color=theme.GREEN,
        height=280,
    )

    reached = metrics.completion_probability(run.cost, budget)
    needed = metrics.budget_for_confidence(run.cost, 0.95)
    left, right = st.columns(2)
    left.metric(
        f"At {theme.money(budget, run.collection.currency, 0)}",
        f"{reached:.1%}",
        delta="chance of finishing",
        delta_color="off",
    )
    right.metric(
        "Budget for 95% confidence",
        theme.money(needed, run.collection.currency, 0),
        delta=f"{needed / budget - 1:+.0%} vs your budget",
        delta_color="inverse",
    )


def render(collection: Collection) -> None:
    theme.page_header(
        "Result Distributions",
        "Histograms of boxes required and total expenditure, with percentile markers.",
    )

    params = params_for(collection)
    run = data.run_simulation(
        params["collection_id"],
        params["num_simulations"],
        params["budget"],
        params["seed"],
    )
    summary = metrics.summarise(run, params["budget"])

    run_caption(run, params["budget"])
    histograms(run)
    percentile_table(run, summary)
    completion_curve(run, params["budget"])
