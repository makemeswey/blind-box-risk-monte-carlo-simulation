# Probabilistic Collection Completion & Cost-Risk Simulator

A Monte Carlo study of what it actually costs to complete a POP MART blind-box
collection, built on three real series: **Jujutsu Kaisen**, **Crybaby** and
**Hirono**.

The POP MART framing is the fun part. Underneath it is a general question about
buying randomised, repeatable items: _how long do you wait, and how much do you
spend, when you cannot choose what you get?_

---

## Background

A blind box is sold sealed. You know the set — twelve regular figures and one
hidden "secret" — but not which one is inside until you open it. Buying the set
is therefore not shopping, it is sampling: every box is an independent draw from
a fixed probability distribution, and you keep drawing until you own one of each.

Two things make that expensive, and neither is obvious from the shelf price:

1. **Duplicates.** The more of the set you own, the less likely a box contains
   something new. Over a full completion run, **91% of boxes are duplicates**.
2. **The secret.** One figure sits at roughly 1-in-144 instead of 1-in-12.
   Completion waits on the _slowest_ figure, so that single figure dominates
   everything else.

Together they turn a "13 × RM 49.80 = RM 647" set into an expected spend of
**RM 7,472**, with a 1-in-20 chance of going past **RM 21,563**.

---

## Overview

The project runs an ETL pipeline into DuckDB, computes an exact analytical
benchmark, simulates tens of thousands of completion runs, and presents the
results in a five-page Streamlit dashboard.

```
data/raw/collections.csv
        ↓  clean  (fractions → decimals, rarity labels, prices)
        ↓  validate  (0 ≤ p ≤ 1, Σp ≈ 1, unique ids, price > 0)
        ↓  load  →  data/processed/figures.parquet  +  database/popmart.duckdb
        ↓
   coupon collector (exact)        Monte Carlo (10k–50k runs)
        ↓                                    ↓
              statistics · tail risk · sensitivity
                            ↓
                   Streamlit dashboard
```

### What the three collections look like

All three currently share the same structure — 12 regular figures at 1/12 and
one secret at 1/144 — so they differ only in price. That is a finding, not an
oversight: the comparison page says so explicitly, and the ranking is driven
purely by box price until the collections differ in size or rarity.

|                        | Jujutsu Kaisen |   Crybaby |    Hirono |
| ---------------------- | -------------: | --------: | --------: |
| Box price              |       RM 62.00 |  RM 49.80 |  RM 49.80 |
| Figures (incl. secret) |             13 |        13 |        13 |
| Expected boxes         |            150 |       150 |       150 |
| Median boxes           |            101 |       101 |       101 |
| P95 boxes              |            433 |       433 |       433 |
| Expected cost          |       RM 9,303 |  RM 7,472 |  RM 7,472 |
| P95 cost               |      RM 26,846 | RM 21,563 | RM 21,563 |
| Duplicate rate         |          91.3% |     91.3% |     91.3% |
| P(complete ≤ RM 5,000) |          42.0% |     49.8% |     49.8% |

### The dashboard

| Page                  | What it answers                                                                   |
| --------------------- | --------------------------------------------------------------------------------- |
| Collection Overview   | What is in the set and at what odds                                               |
| Run Simulation        | Expected boxes, cost, duplicates, and a validation check against the exact result |
| Result Distributions  | The full shape of the outcome — histograms, percentiles, P(complete ≤ budget)     |
| Collection Comparison | Which set is most expensive and most risky                                        |
| Sensitivity Analysis  | What happens when price, rarity or set size changes                               |

---

## How the maths works, in plain terms

### 1. One box is one draw

Each figure `i` has probability `p_i`, and the probabilities sum to 1. Opening a
box means drawing one figure from that distribution. Draws are independent —
POP MART's supply is treated as effectively unlimited, so what you pulled last
time does not change what you pull next.

### 2. Waiting for the next new figure

Suppose you already own `k` of `n` equally likely figures. The chance that the
next box is something new is `(n − k) / n`. When an event has probability `q`,
the average wait for it is `1 / q` tries. So the wait for the next new figure is
`n / (n − k)` boxes.

Add up those waits, from your first box to your last missing figure:

```
E[boxes] = n/n + n/(n−1) + n/(n−2) + … + n/1
         = n × (1 + 1/2 + 1/3 + … + 1/n)
         = n × H(n)
```

That is the classical **coupon collector** result. For 12 equally likely
figures: `12 × H(12) = 37.2 boxes`. The last figure alone accounts for 12 of
them — longer than the 7.8 boxes it took to collect the first six put together.

### 3. What the secret does

The formula above assumes every figure is equally likely. The secret is not: at
1-in-145 it takes about **145 boxes on its own**, and completion cannot finish
before it arrives.

So the expected number of boxes jumps from **37 to 150** — the secret roughly
**quadruples** the whole collection. In a full run the last missing figure alone
takes ~114 boxes on average, about **77% of the entire spend**.

With unequal probabilities there is no `n × H(n)` shortcut. The code uses the
exact identity

```
E[T] = ∫₀^∞ ( 1 − ∏ᵢ (1 − e^(−pᵢ·t)) ) dt
```

evaluated by inclusion–exclusion over subsets of figures, with numerical
integration as a fallback when that sum cancels badly
([`src/probability/coupon_collector.py`](src/probability/coupon_collector.py)).
Read it as: _the collection is incomplete at time `t` if any figure is still
missing; integrating that probability over all `t` gives the average wait._

### 4. Why simulate at all, if the answer is exact?

The formula gives the **average**. It does not tell you how bad an unlucky run
gets, and the average is not what most collectors experience — the median is 101
boxes while the mean is 150, because a long right tail drags the mean up.

So the simulator plays the game instead:

```
repeat 10,000+ times:
    open boxes, one draw at a time, until every figure is owned
    record: boxes, duplicates, total cost
```

From those runs you can read off anything you like: the median, P90, P95, the
worst case, and `P(complete within budget) = share of runs costing ≤ budget`.

### 5. Checking the simulation is right

The exact expectation is the referee. Each run reports how far the simulated
mean sits from it, measured in standard errors (`SE = s / √N`). Within ±2 SE the
simulator is behaving; the Run Simulation page shows this check every time.

### 6. Normalising the odds

The published odds sum to **1.0069**, not 1 — twelve figures at 1/12 plus one at
1/144 slightly overshoots. Every probability is divided by that total so the
distribution is proper, which is why a "1/12" figure appears as **8.28%** rather
than 8.33%, and the secret as 1-in-145 rather than 1-in-144. Both the published
and the normalised values are kept, so nothing is silently rewritten.

---

## Acquiring the data manually

There is no public POP MART API, and the odds are not exposed in any structured
feed. The dataset was therefore **hand-curated** rather than scraped, which is
the honest choice at this size: 39 rows total, and every row can be checked by a
human in minutes.

The whole raw dataset is one CSV,
[`data/raw/collections.csv`](data/raw/collections.csv):

```csv
figure_id,figure_name,rarity,probability,box_price_rm
CRY01,Blossom,regular,1/12,49.8
...
CRY13,Princess Morbucks,secret,1/144,49.8
```

Conventions that make the file work:

- **`figure_id` carries the collection.** The letter prefix (`JJK`, `CRY`,
  `HIR`) is what groups figures into collections — no separate join file to keep
  in sync.
- **Probabilities stay in the form they were published in** (`1/12`, `1/144`).
  The cleaner parses fractions, percentages and decimals, so nothing is
  pre-converted by hand and mis-typed.
- **Prices carry their currency in the column name** (`box_price_rm` → MYR).
- **Adding a collection means adding rows**, nothing else. No simulation code
  changes.

The cleaning and validation steps then do the work a human would otherwise do
badly: standardising rarity labels, converting `1/144` to `0.006944`, stripping
currency symbols, dropping duplicate ids, checking `0 ≤ p ≤ 1`, `Σp ≈ 1` per
collection and `price > 0`. **Invalid rows are flagged, not silently dropped** —
`python -m src.data.load` refuses to load a broken dataset unless you pass
`--allow-invalid`.

### Provenance, stated honestly

Odds were recorded from publicly visible product information for standard
12-piece blind-box series. They are **not** scraped from an authoritative feed,
and the cleaner currently stamps every row with a single default source string.
Treat the 1/12 and 1/144 figures as the documented convention for this series
format rather than a per-product citation — per-row sourcing is the first item
in the next section.

---

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m src.data.load          # clean → validate → parquet → DuckDB
streamlit run main.py            # dashboard on http://localhost:8501
```

Useful command-line entry points, all of which print the same numbers the
dashboard shows:

```bash
python -m src.simulation.simulator --collection CRY --simulations 20000 --budget 5000
python -m src.simulation.simulator --collection all --save      # persist runs to DuckDB
python -m src.analysis.comparison --budget 5000
python -m src.analysis.sensitivity --collection CRY --question --budget 5000
python -m src.analysis.duplicates --collection CRY
```

### Layout

```
main.py                     dashboard shell, sidebar, routing + overview page
src/
  data/         clean · validate · load                   (ETL)
  database/     connection · schema.sql                  (DuckDB)
  probability/  distributions · coupon_collector         (exact maths)
  simulation/   simulator · metrics · experiments        (Monte Carlo)
  analysis/     comparison · sensitivity · duplicates    (results)
  dashboard/    theme · data · one module per page       (UI)
data/raw/collections.csv    the entire hand-curated dataset
database/popmart.duckdb     analytical database
```

---

## Next steps

**Data**

- Per-row source citations (a `source` column in the raw CSV — the cleaner
  already reads one if present) so each probability is traceable to where it was
  seen, rather than to one default string.
- Collections that actually differ in structure — a series with genuine "rare"
  figures, or 6- and 24-piece sets — so the comparison page compares _risk_
  rather than just price.
- Regional pricing (GBP/USD alongside MYR) instead of a single currency.

**Modelling**

- Relax the independence assumption: model a sealed **case** of 12, where the
  distribution is closer to sampling without replacement and duplicates behave
  very differently.
- Add trading and resale — the real completion strategy is rarely "keep opening
  boxes", and a secondary-market price for the secret would change every number
  here.
- Partial-completion goals: what does it cost to reach 10 of 13, which is what
  most collectors actually stop at.

**Engineering**

- Tests for the probability and simulation core (`tests/test_probability.py`,
  `test_simulation.py`, `test_data.py`) — the analytical benchmark makes these
  easy to write and they do not exist yet.
- Notebooks for the write-up (`notebooks/`) and a methodology document
  (`report/methodology.md`).
- Deploy to Streamlit Community Cloud.

---

## Limitations

The model assumes independent draws with constant probabilities, accurate
published odds, a constant box price, unlimited supply, and no trading or
resale. Real collecting violates several of these — cases are sealed in fixed
assortments, prices change, and people swap duplicates. The numbers here
describe _buying single blind boxes at retail until the set is complete_, which
is the worst realistic case and a useful upper bound.

---

## Stack

Python · NumPy · pandas · DuckDB · Streamlit · Altair
