"""Page 5 — what happens to cost and risk when the assumptions move."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis import sensitivity as analysis
from src.dashboard import data, theme
from src.probability.distributions import Collection
from src.simulation.experiments import SECRET_DENOMINATORS

SCENARIO_NOTES = {72: "easier", 144: "base", 288: "harder", 500: "extreme"}

PARAMETER_LABELS = {
    "box_price": "Box price",
    "secret_probability": "Secret probability",
    "collection_size": "Collection size",
    "num_rare": "Number of rare figures",
    "skew": "Probability skew",
}


def published_secret_denominator(collection: Collection) -> int:
    """The odds as printed on the box (1/144), not the normalised 1/145."""
    figures = data.figures_frame()
    secrets = figures[
        (figures["collection_id"] == collection.collection_id)
        & (figures["rarity"] == "secret")
    ]
    if secrets.empty:
        return 144
    return int(round(1 / float(secrets["probability_raw"].min())))


def controls(collection: Collection) -> dict:
    """Modify the collection, then re-price the risk."""
    published_secret = published_secret_denominator(collection)
    denominators = sorted({*SECRET_DENOMINATORS, published_secret})

    with st.form("sensitivity_controls"):
        price, secret, size, sims = st.columns(4)
        box_price = price.number_input(
            f"Box price ({theme.symbol(collection.currency).strip()})",
            min_value=1.0,
            value=float(collection.box_price),
            step=1.0,
        )
        secret_denominator = secret.selectbox(
            "Secret probability",
            options=denominators,
            index=denominators.index(published_secret),
            format_func=lambda d: f"1/{d}",
        )
        num_figures = size.slider(
            "Collection size",
            min_value=4,
            max_value=24,
            value=int(collection.num_figures),
            help="Total figures including the secret.",
        )
        num_simulations = sims.select_slider(
            "Simulations", options=(0, *data.SIMULATION_CHOICES), value=0,
            format_func=lambda n: "analytical only" if n == 0 else f"{n:,}",
        )
        budget = st.number_input(
            f"Budget ({theme.symbol(collection.currency).strip()})",
            min_value=float(collection.box_price),
            value=float(data.default_budget(collection)),
            step=100.0,
        )
        st.form_submit_button("▶  Run Sensitivity", type="primary")

    return {
        "box_price": float(box_price),
        "secret_denominator": int(secret_denominator),
        # 0 leaves the published odds untouched, so "no change" really means no change.
        "secret_override": 0 if secret_denominator == published_secret else int(secret_denominator),
        "num_figures": int(num_figures),
        "num_simulations": int(num_simulations),
        "budget": float(budget),
    }


def variant_metrics(collection: Collection, settings: dict) -> None:
    st.subheader("Your variant vs. the collection as sold")
    baseline = data.analytical_summary(collection.collection_id, settings["budget"])
    variant = data.variant_summary(
        collection.collection_id,
        settings["box_price"],
        settings["secret_override"],
        settings["num_figures"],
        settings["budget"],
        settings["num_simulations"],
    )
    currency = collection.currency

    boxes, cost, tail, completion = st.columns(4)
    boxes.metric(
        "Expected boxes",
        f"{variant['expected_boxes']:.0f}",
        delta=f"{variant['expected_boxes'] - baseline['expected_boxes']:+.0f}",
        delta_color="inverse",
    )
    cost.metric(
        "Expected cost",
        theme.money(variant["expected_cost"], currency, 0),
        delta=f"{variant['expected_cost'] / baseline['expected_cost'] - 1:+.0%}",
        delta_color="inverse",
    )
    tail.metric(
        "P95 cost",
        theme.money(variant["p95_cost"], currency, 0),
        delta=f"{variant['p95_cost'] / baseline['p95_cost'] - 1:+.0%}",
        delta_color="inverse",
    )
    completion.metric(
        f"P(≤ {theme.money(settings['budget'], currency, 0)})",
        f"{variant['completion_probability']:.1%}",
        delta=f"{variant['completion_probability'] - baseline['completion_probability']:+.1%}",
    )

    if settings["num_simulations"]:
        st.caption(
            f"Monte Carlo cross-check over {settings['num_simulations']:,} runs: "
            f"{variant['sim_expected_boxes']:.1f} boxes, "
            f"{theme.money(variant['sim_expected_cost'], currency, 0)} expected, "
            f"{theme.money(variant['sim_p95_cost'], currency, 0)} at P95."
        )


def scenario_cards(frame: pd.DataFrame, settings: dict, currency: str) -> None:
    st.subheader("Secret rarity impact")
    scenarios = frame[frame["scenario"].str.startswith("secret ")]

    for column, row in zip(st.columns(len(scenarios)), scenarios.itertuples()):
        denominator = int(round(row.secret_odds_1_in))
        with column, st.container(border=True):
            st.markdown(f"**1/{denominator}** — {SCENARIO_NOTES.get(denominator, 'variant')}")
            st.metric("Expected boxes", f"{row.expected_boxes:.0f}")
            st.metric("Expected cost", theme.money(row.expected_cost, currency, 0))
            st.metric("P95 cost", theme.money(row.p95_cost, currency, 0))
            st.metric(
                f"P(≤ {theme.money(settings['budget'], currency, 0)})",
                f"{row.completion_probability:.0%}",
            )

    no_secret = frame.loc[frame["scenario"] == "no secret"].iloc[0]
    published = frame.loc[frame["scenario"] == "as published"].iloc[0]
    st.caption(
        f"Without any secret the same set costs "
        f"{theme.money(no_secret.expected_cost, currency, 0)} on average — the single "
        f"secret figure multiplies expected spend by "
        f"{published.expected_cost / no_secret.expected_cost:.1f}×."
    )


def secret_cost_chart(frame: pd.DataFrame, currency: str) -> None:
    st.subheader("Expected cost vs. secret figure probability")
    scenarios = frame[frame["scenario"].str.startswith("secret ")].assign(
        odds=lambda f: f["secret_odds_1_in"].round().astype(int)
    ).rename(columns={"expected_cost": "Expected cost", "p95_cost": "P95 cost"})

    st.line_chart(
        scenarios,
        x="odds",
        y=["Expected cost", "P95 cost"],
        x_label="Secret rarity (1 in N)",
        y_label=f"Cost ({currency})",
        color=[theme.ACCENT, theme.BLUE],
        height=280,
    )


def budget_answer(collection: Collection, settings: dict) -> None:
    st.subheader("Worked question")
    impact = data.secret_change_impact(
        collection.collection_id, settings["budget"], 144, 288
    )
    currency = collection.currency
    st.info(
        f"Halving the secret's odds from 1/144 to 1/288 moves the chance of "
        f"completing {collection.collection_name} within "
        f"{theme.money(impact['budget'], currency, 0)} "
        f"({impact['boxes_affordable']} boxes) from "
        f"{impact['completion_before']:.1%} to {impact['completion_after']:.1%}, and "
        f"multiplies expected spend by {impact['cost_multiplier']:.2f}× "
        f"({theme.money(impact['expected_cost_before'], currency, 0)} → "
        f"{theme.money(impact['expected_cost_after'], currency, 0)})."
    )


def sweep_explorer(collection: Collection, settings: dict) -> None:
    st.subheader("One-parameter sweep")
    parameter = st.segmented_control(
        "Parameter",
        options=list(PARAMETER_LABELS),
        format_func=PARAMETER_LABELS.get,
        default="secret_probability",
        key="sweep_parameter",
    )
    if parameter is None:
        st.caption("Pick a parameter to sweep.")
        return

    values = tuple(float(v) for v in analysis.default_values(collection, parameter))
    frame = data.sweep_frame(
        collection.collection_id, parameter, values, settings["budget"]
    )
    currency = collection.currency

    display = frame[
        [
            "value",
            "num_figures",
            "expected_boxes",
            "p95_boxes",
            "expected_cost",
            "p95_cost",
            "duplicate_rate",
            "completion_probability",
            "cost_vs_baseline",
        ]
    ].rename(
        columns={
            "value": PARAMETER_LABELS[parameter],
            "num_figures": "Figures",
            "expected_boxes": "Expected boxes",
            "p95_boxes": "P95 boxes",
            "expected_cost": f"Expected cost ({currency})",
            "p95_cost": f"P95 cost ({currency})",
            "duplicate_rate": "Duplicate rate",
            "completion_probability": "P(complete)",
            "cost_vs_baseline": "vs baseline",
        }
    )
    if parameter == "secret_probability":
        display[PARAMETER_LABELS[parameter]] = (1 / frame["value"]).round().astype(int)
        display = display.rename(
            columns={PARAMETER_LABELS[parameter]: "Secret rarity (1 in N)"}
        )
    display = display.round(
        {
            "Expected boxes": 1,
            "P95 boxes": 0,
            f"Expected cost ({currency})": 0,
            f"P95 cost ({currency})": 0,
        }
    )

    money = st.column_config.NumberColumn(format="localized")
    st.dataframe(
        display,
        hide_index=True,
        column_config={
            "Expected boxes": st.column_config.NumberColumn(format="%.1f"),
            "P95 boxes": st.column_config.NumberColumn(format="%.0f"),
            f"Expected cost ({currency})": money,
            f"P95 cost ({currency})": money,
            "Duplicate rate": st.column_config.NumberColumn(format="percent"),
            "P(complete)": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0
            ),
            "vs baseline": st.column_config.NumberColumn(format="%.2fx"),
        },
    )

    slope = analysis.elasticity(frame)
    st.caption(
        f"Log-log slope of expected cost against {PARAMETER_LABELS[parameter].lower()}: "
        f"{slope:.2f} — values above 1 mean the parameter is amplified, below 1 damped."
    )


def render(collection: Collection) -> None:
    theme.page_header(
        "Sensitivity Analysis",
        "See how changing price, rarity and collection size affects completion cost and risk.",
    )

    settings = controls(collection)
    variant_metrics(collection, settings)

    experiment = data.secret_experiment(
        collection.collection_id, SECRET_DENOMINATORS, settings["budget"]
    )
    scenario_cards(experiment, settings, collection.currency)
    secret_cost_chart(experiment, collection.currency)
    budget_answer(collection, settings)
    sweep_explorer(collection, settings)
