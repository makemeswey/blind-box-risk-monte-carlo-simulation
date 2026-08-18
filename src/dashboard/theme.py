"""Look and feel shared by every page.

The CSS only restyles Streamlit's own components (sidebar, metrics, tables,
buttons) so the pages can be built from `st.*` widgets rather than hand-written
HTML.
"""

from __future__ import annotations

from fractions import Fraction

import streamlit as st

CURRENCY_SYMBOLS = {"MYR": "RM ", "GBP": "£", "USD": "$", "EUR": "€"}

ACCENT = "#ff4b6e"
BLUE = "#636efa"
GREEN = "#00cc96"
ORANGE = "#ffa15a"
PURPLE = "#ab63fa"
RED = "#c62828"

RARITY_COLOURS = {"regular": BLUE, "rare": PURPLE, "secret": ACCENT}

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
  --border: #e5e7eb;
  --metric-bg: #f0f2f6;
}

html, body, .stApp, button, input, select, textarea {
  font-family: 'Source Sans 3', -apple-system, sans-serif;
}
/* keep Streamlit's icon ligatures on their own font */
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: var(--main-bg); color: var(--text-primary); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding: 2.2rem 3rem 4rem; }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
  background: var(--sidebar-bg); width: 268px !important; border-right: none;
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
.sidebar-footer {
  border-top: 1px solid #21262d; padding-top: 14px; margin-top: 8px;
  font-size: 11px; color: #484f58 !important; line-height: 1.6;
}
.sidebar-footer span { color: var(--accent) !important; }

[data-testid="stSidebar"] label p { font-size: 13px; font-weight: 600; color: #c9d1d9; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
  background: #161b22; border: 1px solid #30363d; border-radius: 6px;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
  color: #fff; font-size: 13px; font-weight: 500;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg { fill: #8b949e; color: #8b949e; }

/* Nav — radio rendered as the mockup's button list */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
  display: flex; width: 100%; padding: 9px 12px; margin: 0; border-radius: 6px;
  transition: background 0.15s, color 0.15s;
}
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

/* ─── Page header ─── */
.block-container h1 {
  font-size: 28px; font-weight: 700; color: var(--text-primary);
  padding: 0; margin: 6px 0 4px;
}
.block-container h2 {
  font-size: 18px; font-weight: 700; color: var(--text-primary);
  padding: 0 0 8px; margin: 22px 0 10px; border-bottom: 2px solid var(--border);
}
.block-container h3 { font-size: 15px; font-weight: 700; padding: 0; margin: 6px 0; }

/* ─── Metrics as the mockup's cards ─── */
[data-testid="stMetric"] {
  background: var(--metric-bg); border: none; border-radius: 8px; padding: 16px 18px;
}
[data-testid="stMetricLabel"] p {
  font-size: 12px !important; font-weight: 600; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
  font-size: 26px; font-weight: 700; font-family: 'JetBrains Mono', monospace;
  color: var(--text-primary);
}
[data-testid="stMetricDelta"] { font-size: 12px; font-weight: 600; }

/* Cards used for scenario blocks */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetric"] {
  background: transparent; padding: 2px 0;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricValue"] { font-size: 20px; }

/* ─── Tables, charts, controls ─── */
[data-testid="stDataFrame"] { border-radius: 8px; }
[data-testid="stElementToolbar"] { background: transparent; }

.stButton button, .stFormSubmitButton button {
  border-radius: 6px; font-weight: 600; padding: 8px 26px;
}
[data-testid="stForm"] {
  border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 20px 4px; background: var(--card-bg);
}
</style>
"""


def inject() -> None:
    """Apply the shared stylesheet — call once per rerun, before anything renders."""
    st.markdown(STYLES, unsafe_allow_html=True)


def symbol(currency: str) -> str:
    return CURRENCY_SYMBOLS.get(currency, currency + " ")


def money(amount: float, currency: str = "MYR", decimals: int = 2) -> str:
    return f"{symbol(currency)}{amount:,.{decimals}f}"


def format_odds(probability: float) -> str:
    """Render a probability as a `1/n` style fraction (0.006944 → '1/144')."""
    if probability is None or probability <= 0:
        return "—"
    fraction = Fraction(probability).limit_denominator(10_000)
    if fraction.numerator == 1:
        return f"1/{fraction.denominator}"
    return f"1 in {1 / probability:,.1f}"


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
