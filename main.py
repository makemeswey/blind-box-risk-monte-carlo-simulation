"""POP MART Monte Carlo Collection Simulator — Streamlit dashboard.

    streamlit run main.py

This module is the shell (sidebar, routing) plus page 1; pages 2–5 live in
`src/dashboard/`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import comparison, data, distribution, sensitivity, simulation, theme
from src.probability.distributions import Collection

PAGES = {
    "📦 Overview": None,  # rendered below by render_overview
    "🎲 Simulation": simulation.render,
    "📊 Distribution": distribution.render,
    "⚖️ Collection Comparison": comparison.render,
    "🔬 Sensitivity Analysis": sensitivity.render,
}


# ─────────────────────────── page 1 — overview ───────────────────────────

def overview_metrics(figures: pd.DataFrame, collection: Collection) -> None:
    secrets = figures[figures["rarity"] == "secret"]
    ordinary = len(figures) - len(secrets)
    secret_odds = (
        theme.format_odds(float(secrets["probability_raw"].min())) if len(secrets) else "—"
    )

    count, secret, price, odds = st.columns(4)
    count.metric("Figures", ordinary, help="Regular and rare figures, excluding secrets.")
    secret.metric("+ Secret", len(secrets))
    price.metric("Box price", theme.money(collection.box_price, collection.currency))
    odds.metric("Secret odds", secret_odds, help="As published on the packaging.")


def probability_table(figures: pd.DataFrame) -> None:
    st.subheader("Figure probabilities")
    table = pd.DataFrame(
        {
            "Figure": figures["figure_name"],
            "Rarity": figures["rarity"].str.capitalize(),
            "Published odds": figures["probability_raw"].map(theme.format_odds),
            "Probability": figures["probability"],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        height=(len(table) + 1) * 35 + 3,  # every figure visible, no inner scroll
        column_config={"Probability": st.column_config.NumberColumn(format="percent")},
    )


def source_note(figures: pd.DataFrame) -> None:
    published = float(figures["probability_raw"].sum())
    normalised = float(figures["probability"].sum())
    st.caption(
        f"Published odds sum to {published:.4f}; probabilities are normalised to "
        f"{normalised:.4f} so the simulator samples from a proper distribution. "
        f"Source: {figures['source'].iloc[0]}"
    )


def render_overview(collection: Collection, figures: pd.DataFrame) -> None:
    theme.page_header(
        "Collection Overview",
        "Explore the selected collection's figures, probabilities, and rarity structure.",
    )
    overview_metrics(figures, collection)
    probability_table(figures)
    source_note(figures)


# ─────────────────────────── shell ───────────────────────────

def render_sidebar(figures: pd.DataFrame) -> tuple[str, str]:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-logo">BLIND BOX</div>'
            '<div class="sidebar-title">Monte Carlo Collection Risk Simulator</div>',
            unsafe_allow_html=True,
        )

        names = (
            figures[["collection_id", "collection_name"]]
            .drop_duplicates()
            .sort_values("collection_name")
        )
        name_to_id = dict(zip(names["collection_name"], names["collection_id"]))
        collection_name = st.selectbox("Collection", list(name_to_id))

        st.markdown('<div class="sidebar-label">Pages</div>', unsafe_allow_html=True)
        page = st.radio("Pages", list(PAGES), label_visibility="collapsed")

        st.markdown(
            '<div class="sidebar-footer">'
            "This is just a fun project, please do not use it to influence any financial decisons :D"
            "</div>",
            unsafe_allow_html=True,
        )

    return name_to_id[collection_name], page


def main() -> None:
    st.set_page_config(
        page_title="Blind Box Risk Monte Carlo Simulator",
        page_icon="🎲",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.inject()

    try:
        figures = data.figures_frame()
        collections = data.all_collections()
    except Exception as error:
        st.error(f"Could not read the database: {error}")
        st.caption("Run `python -m src.data.load` to build it, then reload this page.")
        return

    collection_id, page = render_sidebar(figures)
    collection = collections[collection_id]

    if PAGES[page] is None:
        render_overview(collection, figures[figures["collection_id"] == collection_id])
    else:
        PAGES[page](collection)


if __name__ == "__main__":
    main()
