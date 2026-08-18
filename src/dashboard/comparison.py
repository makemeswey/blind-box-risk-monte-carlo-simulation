"""Page 4 — JJK vs Crybaby vs Hirono, head to head."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.analysis import comparison as analysis
from src.dashboard import data, theme
from src.probability.distributions import Collection

PALETTE = (theme.BLUE, theme.ACCENT, theme.GREEN, theme.PURPLE, theme.ORANGE)

BAR_TABS = {
    "Expected cost": ("expected_cost", "money"),
    "P95 tail risk": ("p95_cost", "money"),
    "Completion probability": ("completion_probability", "percent"),
    "Duplicate rate": ("duplicate_rate", "percent"),
}


def controls(collection: Collection) -> tuple[float, int]:
    budget, cross_check, simulations = st.columns([2, 1, 2])
    amount = budget.number_input(
        f"Budget ({theme.symbol(collection.currency).strip()})",
        min_value=float(collection.box_price),
        value=float(data.default_budget(collection)),
        step=100.0,
        key="comparison_budget",
    )
    with cross_check:
        st.write("")
        enabled = st.toggle("Cross-check", value=False, help="Add Monte Carlo columns beside the exact ones.")
    num_simulations = simulations.select_slider(
        "Simulations per collection",
        options=data.SIMULATION_CHOICES,
        value=10_000,
        disabled=not enabled,
    )
    return float(amount), int(num_simulations) if enabled else 0


def key_metrics_table(frame: pd.DataFrame, currency: str) -> None:
    st.subheader("Key metrics")
    matrix = analysis.comparison_matrix(frame)
    st.dataframe(format_matrix(matrix, currency), height=(len(matrix) + 1) * 35 + 3)


def format_matrix(matrix: pd.DataFrame, currency: str) -> pd.DataFrame:
    """One column per collection; each metric row carries its own unit."""
    formatted = matrix.copy().astype(object)
    for label in matrix.index:
        for column in matrix.columns:
            value = float(matrix.loc[label, column])
            lowered = label.lower()
            if "cost" in lowered or "price" in lowered or "budget" in lowered:
                formatted.loc[label, column] = theme.money(value, currency)
            elif "rate" in lowered or "probability" in lowered:
                formatted.loc[label, column] = f"{value:.1%}"
            elif value == int(value):
                formatted.loc[label, column] = f"{int(value):,}"
            else:
                formatted.loc[label, column] = f"{value:,.1f}"
    return formatted


def bar_chart(frame: pd.DataFrame, column: str, unit: str, currency: str) -> alt.Chart:
    names = list(frame["collection_name"])
    value_format = ".1%" if unit == "percent" else ",.0f"
    axis_format = ".0%" if unit == "percent" else ",.0f"
    label = f"{column.replace('_', ' ').capitalize()}"
    if unit == "money":
        label = f"{label} ({currency})"

    base = alt.Chart(frame).encode(
        y=alt.Y(
            "collection_name:N",
            title=None,
            sort=names,
            axis=alt.Axis(labelLimit=220, labelFontSize=13),
        ),
        x=alt.X(f"{column}:Q", title=label, axis=alt.Axis(format=axis_format)),
        color=alt.Color(
            "collection_name:N",
            scale=alt.Scale(domain=names, range=list(PALETTE[: len(names)])),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("collection_name:N", title="Collection"),
            alt.Tooltip(f"{column}:Q", title=label, format=value_format),
        ],
    )
    labels = base.mark_text(align="left", dx=6, fontWeight="bold", color="#1a1a2e").encode(
        text=alt.Text(f"{column}:Q", format=value_format)
    )
    return (base.mark_bar(cornerRadius=4) + labels).properties(height=64 * len(names))


def comparison_charts(frame: pd.DataFrame, currency: str) -> None:
    st.subheader("Cost and risk, side by side")
    for tab, (title, (column, unit)) in zip(st.tabs(list(BAR_TABS)), BAR_TABS.items()):
        with tab:
            st.altair_chart(bar_chart(frame, column, unit, currency), width="stretch")


def risk_ranking(frame: pd.DataFrame) -> None:
    st.subheader("Risk ranking")
    ranking = analysis.risk_ranking(frame).rename(
        columns={
            "collection_name": "Collection",
            "by_expected_cost": "Expected cost",
            "by_p95_cost": "P95 cost",
            "by_expected_boxes": "Expected boxes",
            "by_completion_risk": "Completion risk",
        }
    )
    st.dataframe(ranking, hide_index=True)
    st.caption("Rank 1 is the worst outcome for the collector on that dimension.")


def verdict(frame: pd.DataFrame) -> None:
    st.subheader("Which collection is riskiest?")
    # two-space suffix keeps each finding on its own line inside the callout
    st.info(
        analysis.verdict(frame, list(data.all_collections().values())).replace("\n", "  \n")
    )


def render(collection: Collection) -> None:
    theme.page_header(
        "Collection Comparison",
        "Head-to-head comparison of completion cost and risk across all three collections.",
    )

    budget, num_simulations = controls(collection)
    frame = data.comparison_frame(budget, num_simulations)
    currency = collection.currency

    key_metrics_table(frame, currency)
    comparison_charts(frame, currency)
    risk_ranking(frame)
    verdict(frame)

    if num_simulations:
        st.subheader("Monte Carlo cross-check")
        simulated = frame[
            ["collection_name", "expected_cost", "sim_expected_cost", "p95_cost", "sim_p95_cost"]
        ].rename(
            columns={
                "collection_name": "Collection",
                "expected_cost": f"Expected cost, exact ({currency})",
                "sim_expected_cost": f"Expected cost, simulated ({currency})",
                "p95_cost": f"P95 cost, exact ({currency})",
                "sim_p95_cost": f"P95 cost, simulated ({currency})",
            }
        )
        money_column = st.column_config.NumberColumn(format="localized")
        st.dataframe(
            simulated,
            hide_index=True,
            column_config={c: money_column for c in simulated.columns if c != "Collection"},
        )
