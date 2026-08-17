from __future__ import annotations
import html
from fractions import Fraction
from pathlib import Path
import pandas as pd
import streamlit as st

from src.database.connection import DB_PATH, get_connection

PAGES = [
    "📦 Overview",
    "🎲 Simulation",
    "📊 Distribution",
    "⚖️ Collection Comparison",
    "🔬 Sensitivity Analysis",
]

CURRENCY_SYMBOLS = {"MYR": "RM", "GBP": "£", "USD": "$", "EUR": "€"}

RARITY_COLOURS = {
    "regular": "var(--blue)",
    "rare": "var(--purple)",
    "secret": "var(--accent)",
}

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --sidebar-bg: #0e1117;
  --sidebar-hover: #1a1f2e;
  --main-bg: #ffffff;
  --card-bg: #f8f9fb;
  --text-primary: #1a1a2e;
  --text-secondary: #6b7280;
  --accent: #ff4b6e;
  --accent-soft: #fff0f3;
  --blue: #636efa;
  --green: #00cc96;
  --orange: #ffa15a;
  --purple: #ab63fa;
  --border: #e5e7eb;
  --metric-bg: #f0f2f6;
}

html, body, [class*="st-"] { font-family: 'Source Sans 3', -apple-system, sans-serif; }
.stApp { background: var(--main-bg); color: var(--text-primary); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding: 2.2rem 3rem 4rem; }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
  background: var(--sidebar-bg);
  width: 268px !important;
  border-right: none;
}
[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
[data-testid="stSidebar"] * { color: #c9d1d9; }

.sidebar-logo {
  font-size: 13px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--accent) !important; margin-bottom: 4px;
}
.sidebar-title {
  font-size: 18px; font-weight: 700; line-height: 1.3;
  color: #fff !important; margin-bottom: 4px;
}
.sidebar-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; color: #6b7280 !important; margin: 6px 0 2px;
}

/* Collection picker */
[data-testid="stSidebar"] label p { font-size: 13px; font-weight: 600; color: #c9d1d9; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
  background: #161b22; border: 1px solid #30363d; border-radius: 6px;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
  color: #fff; font-size: 13px; font-weight: 500;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg { fill: #8b949e; color: #8b949e; }

/* Nav (radio styled as buttons) */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
  display: flex; width: 100%; padding: 9px 12px; margin: 0; border-radius: 6px;
  transition: background 0.15s, color 0.15s;
}
/* hide the radio glyph — the whole row is the button */
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child {
  display: none;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {
  font-size: 14px; font-weight: 400; color: #c9d1d9;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background: var(--sidebar-hover); }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover p { color: #fff; }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
  background: rgba(255, 75, 110, 0.12);
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
  color: var(--accent); font-weight: 600;
}

.sidebar-footer {
  border-top: 1px solid #21262d; padding-top: 14px; margin-top: 8px;
  font-size: 11px; color: #484f58 !important; line-height: 1.6;
}
.sidebar-footer span { color: var(--accent) !important; }

/* ─── Page header ─── */
.page-badge {
  display: inline-block; font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--accent); background: var(--accent-soft);
  padding: 4px 10px; border-radius: 4px; margin-bottom: 10px;
}
h1.page-title {
  font-size: 28px; font-weight: 700; color: var(--text-primary);
  margin: 0 0 6px; padding: 0;
}
.page-subtitle {
  font-size: 15px; color: var(--text-secondary); line-height: 1.5; margin-bottom: 28px;
}
h2.section-header {
  font-size: 18px; font-weight: 700; color: var(--text-primary);
  margin: 8px 0 16px; padding-bottom: 8px; border-bottom: 2px solid var(--border);
}

/* ─── Metric cards ─── */
.metrics-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin-bottom: 28px;
}
.metric-card { background: var(--metric-bg); border-radius: 8px; padding: 16px 18px; }
.metric-card .label {
  font-size: 12px; font-weight: 600; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;
}
.metric-card .value {
  font-size: 26px; font-weight: 700; color: var(--text-primary);
  font-family: 'JetBrains Mono', monospace; line-height: 1.2;
}
.metric-card .value.small { font-size: 20px; }
.metric-card .caption { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }

/* ─── Table ─── */
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px; }
.data-table th {
  text-align: left; font-weight: 700; font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.04em; color: var(--text-secondary); padding: 10px 14px;
  border-bottom: 2px solid var(--border); background: var(--card-bg);
}
.data-table td {
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--text-primary);
}
.data-table td.name { font-family: 'Source Sans 3', sans-serif; font-weight: 600; }
.data-table tr:hover td { background: var(--card-bg); }

.rarity-badge {
  display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px;
  border-radius: 4px; text-transform: uppercase; letter-spacing: 0.03em;
  font-family: 'Source Sans 3', sans-serif;
}
.rarity-badge.regular { background: #e8f5e9; color: #2e7d32; }
.rarity-badge.rare    { background: #e3f2fd; color: #1565c0; }
.rarity-badge.secret  { background: #fce4ec; color: #c62828; }

.prob-bar-cell { display: flex; align-items: center; gap: 8px; }
.prob-bar-cell .bar-track {
  flex: 1; height: 8px; background: var(--border); border-radius: 4px;
  overflow: hidden; max-width: 140px;
}
.prob-bar-cell .bar-fill { height: 100%; border-radius: 4px; }

.note {
  font-size: 12px; color: var(--text-secondary); line-height: 1.6;
  border-left: 3px solid var(--border); padding: 2px 0 2px 12px; margin-bottom: 24px;
}

.stub {
  background: var(--card-bg); border: 1px dashed var(--border); border-radius: 10px;
  padding: 36px; text-align: center; color: var(--text-secondary); font-size: 14px;
}
</style>
"""


# ─────────────────────────── data ───────────────────────────

@st.cache_data(show_spinner=False)
def load_figures(db_path: str = str(DB_PATH)) -> pd.DataFrame:
    conn = get_connection(Path(db_path), read_only=True)
    try:
        return conn.execute(
            """
            SELECT collection_id, collection_name, figure_id, figure_name, rarity,
                   probability_raw, probability, box_price, currency, source
            FROM figures
            ORDER BY collection_id, figure_id
            """
        ).df()
    finally:
        conn.close()


# ─────────────────────────── formatting ───────────────────────────

def format_odds(probability: float) -> str:
    """Render a probability as a `1/n` style fraction (0.006944 → '1/144')."""
    if probability <= 0:
        return "—"
    fraction = Fraction(probability).limit_denominator(10_000)
    if fraction.numerator == 1:
        return f"1/{fraction.denominator}"
    return f"1 in {1 / probability:,.1f}"


def format_price(price: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{symbol}{price:,.2f}"


def metric_card(label: str, value: str, caption: str | None = None, small: bool = False) -> str:
    caption_html = f'<div class="caption">{html.escape(caption)}</div>' if caption else ""
    value_class = "value small" if small else "value"
    return (
        '<div class="metric-card">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="{value_class}">{html.escape(value)}</div>'
        f"{caption_html}</div>"
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<p class="page-subtitle">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )

def render_overview(figures: pd.DataFrame) -> None:
    page_header(
        "Collection Overview",
        "Explore the selected collection's figures, probabilities, and rarity structure.",
    )

    currency = figures["currency"].iloc[0]
    box_price = float(figures["box_price"].iloc[0])
    secrets = figures[figures["rarity"] == "secret"]
    rares = figures[figures["rarity"] == "rare"]
    regulars = figures[figures["rarity"] == "regular"]

    secret_odds = format_odds(float(secrets["probability_raw"].min())) if not secrets.empty else "—"

    cards = [
        metric_card("Figures", str(len(regulars) + len(rares)), "regular + rare"),
        metric_card("+ Secret", str(len(secrets))),
        metric_card("Box Price", format_price(box_price, currency)),
        metric_card("Secret Odds", secret_odds, "as published", small=True),
    ]
    st.markdown(f'<div class="metrics-row">{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown('<h2 class="section-header">Figure Probabilities</h2>', unsafe_allow_html=True)
    st.markdown(probability_table(figures), unsafe_allow_html=True)

    total = float(figures["probability"].sum())
    raw_total = float(figures["probability_raw"].sum())
    source = str(figures["source"].iloc[0])
    st.markdown(
        '<div class="note">'
        f"Published odds sum to {raw_total:.4f}; probabilities are normalised to "
        f"{total:.4f} so the simulator samples from a proper distribution.<br>"
        f"Source: {html.escape(source)}"
        "</div>",
        unsafe_allow_html=True,
    )


def probability_table(figures: pd.DataFrame) -> str:
    max_probability = float(figures["probability"].max())
    rows = []
    for figure in figures.itertuples():
        colour = RARITY_COLOURS.get(figure.rarity, "var(--blue)")
        bar_width = figure.probability / max_probability * 100
        prefix = "⭐ " if figure.rarity == "secret" else ""
        rows.append(
            "<tr>"
            f'<td class="name">{prefix}{html.escape(str(figure.figure_name))}</td>'
            f'<td><span class="rarity-badge {html.escape(figure.rarity)}">'
            f"{html.escape(figure.rarity)}</span></td>"
            f"<td>{format_odds(figure.probability_raw)}</td>"
            f"<td>{figure.probability * 100:.2f}%</td>"
            '<td><div class="prob-bar-cell"><div class="bar-track">'
            f'<div class="bar-fill" style="width:{bar_width:.1f}%;background:{colour}"></div>'
            "</div></div></td>"
            "</tr>"
        )

    return (
        '<table class="data-table"><thead><tr>'
        "<th>Figure</th><th>Rarity</th><th>Published Odds</th>"
        "<th>Probability</th><th>Distribution</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_stub(page: str, badge: str, title: str, subtitle: str) -> None:
    page_header(badge, title, subtitle)
    st.markdown(
        f'<div class="stub">{html.escape(page)} is not built yet.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────── app ───────────────────────────

def render_sidebar(figures: pd.DataFrame) -> tuple[str, str]:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-logo">POP MART</div>'
            '<div class="sidebar-title">Monte Carlo Collection Simulator</div>',
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
        page = st.radio("Pages", PAGES, label_visibility="collapsed")


    return name_to_id[collection_name], page


def main() -> None:
    st.set_page_config(
        page_title="POP MART Monte Carlo Simulator",
        page_icon="🎲",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(STYLES, unsafe_allow_html=True)

    try:
        figures = load_figures()
    except Exception as error:
        st.error(f"Could not read {DB_PATH}: {error}\n\nRun `python -m src.data.load` first.")
        return

    collection_id, page = render_sidebar(figures)
    selected = figures[figures["collection_id"] == collection_id]

    if page == PAGES[0]:
        render_overview(selected)
    elif page == PAGES[1]:
        render_stub(page, "Page 2", "Run Simulation",
                    "Configure and run a Monte Carlo simulation to estimate collection completion cost.")
    elif page == PAGES[2]:
        render_stub(page, "Page 3", "Result Distributions",
                    "Histograms of boxes required and total expenditure, with percentile markers.")
    elif page == PAGES[3]:
        render_stub(page, "Page 4", "Collection Comparison",
                    "Head-to-head comparison of completion cost and risk across all three collections.")
    else:
        render_stub(page, "Page 5", "Sensitivity Analysis",
                    "See how changing parameters affects completion cost and risk.")


if __name__ == "__main__":
    main()
