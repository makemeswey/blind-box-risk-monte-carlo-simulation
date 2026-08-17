from __future__ import annotations
import itertools
import math
from math import comb, prod
import numpy as np
from src.probability.distributions import Collection, load_collections

MAX_TERMS = 2_000_000
GROUP_DECIMALS = 12
CANCELLATION_LIMIT = 1e-9
QUADRATURE_NODES = 4000

def harmonic_number(n: int) -> float:
    return float(np.sum(1 / np.arange(1, n + 1)))


def expected_boxes_uniform(n: int) -> float:
    return n * harmonic_number(n)


def variance_boxes_uniform(n: int) -> float:
    k = np.arange(1, n + 1)
    return float(n**2 * np.sum(1 / k**2) - n * harmonic_number(n))


def std_boxes_uniform(n: int) -> float:
    return float(np.sqrt(variance_boxes_uniform(n)))

def _groups(probabilities) -> tuple[np.ndarray, np.ndarray]:
    values, counts = np.unique(
        np.round(np.asarray(probabilities, dtype=float), GROUP_DECIMALS),
        return_counts=True,
    )
    num_terms = prod(int(c) + 1 for c in counts)
    if num_terms > MAX_TERMS:
        raise ValueError(
            f"{num_terms:,} inclusion-exclusion terms exceeds MAX_TERMS "
            f"({MAX_TERMS:,}); use the Monte Carlo estimate instead"
        )
    return values, counts


def _subset_terms(probabilities):
    values, counts = _groups(probabilities)
    for taken in itertools.product(*(range(int(c) + 1) for c in counts)):
        size = sum(taken)
        if size == 0:
            continue
        multiplicity = prod(comb(int(c), k) for c, k in zip(counts, taken))
        p_subset = float(np.dot(taken, values))
        yield size, multiplicity, p_subset


def _alternating_sum(terms) -> tuple[float, float]:
    terms = list(terms)
    return math.fsum(terms), max(abs(term) for term in terms)


def _is_reliable(total: float, largest_term: float) -> bool:
    return abs(largest_term) * np.finfo(float).eps <= CANCELLATION_LIMIT * abs(total)

def _expected_boxes_quadrature(probabilities, nodes: int = QUADRATURE_NODES) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    p_min = probabilities.min()

    x, w = np.polynomial.legendre.leggauss(nodes)
    v = 0.5 * (x + 1)
    w = 0.5 * w

    t = -np.log1p(-v) / p_min
    dt_dv = 1 / (p_min * (1 - v))
    survival = 1 - np.prod(1 - np.exp(-np.outer(t, probabilities)), axis=1)
    return float(np.dot(w, survival * dt_dv))


def expected_boxes(probabilities, method: str = "auto") -> float:
    if method == "quadrature":
        return _expected_boxes_quadrature(probabilities)

    total, largest = _alternating_sum(
        (-1) ** (size + 1) * multiplicity / p_subset
        for size, multiplicity, p_subset in _subset_terms(probabilities)
    )
    if method == "exact" or _is_reliable(total, largest):
        return total
    return _expected_boxes_quadrature(probabilities)


def variance_boxes(probabilities) -> float:
    second_moment, largest = _alternating_sum(
        (-1) ** (size + 1)
        * multiplicity
        * (2 * (1 - p_subset) / p_subset**2 + 1 / p_subset)
        for size, multiplicity, p_subset in _subset_terms(probabilities)
    )
    mean = expected_boxes(probabilities)
    if not _is_reliable(second_moment, largest):
        raise ValueError(
            "Inclusion-exclusion for Var[T] is numerically unreliable for this "
            "distribution; use the Monte Carlo estimate instead"
        )
    return float(second_moment - mean**2)


def std_boxes(probabilities) -> float:
    return float(np.sqrt(variance_boxes(probabilities)))

def completion_probability(probabilities, boxes: int) -> float:
    if boxes < len(np.asarray(probabilities)):
        return 0.0

    survival, largest = _alternating_sum(
        (-1) ** (size + 1) * multiplicity * max(0.0, 1.0 - p_subset) ** boxes
        for size, multiplicity, p_subset in _subset_terms(probabilities)
    )

    if abs(largest) * np.finfo(float).eps > CANCELLATION_LIMIT:
        raise ValueError(
            f"Inclusion-exclusion for P(T <= {boxes}) is numerically unreliable "
            "for this distribution; use the Monte Carlo estimate instead"
        )
    return float(min(1.0, max(0.0, 1.0 - survival)))


def boxes_for_quantile(probabilities, quantile: float) -> int:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1")

    low = len(np.asarray(probabilities))
    high = max(low, 1)
    while completion_probability(probabilities, high) < quantile:
        low, high = high, high * 2

    while low < high:
        mid = (low + high) // 2
        if completion_probability(probabilities, mid) >= quantile:
            high = mid
        else:
            low = mid + 1
    return int(low)

def analytical_summary(collection: Collection, budget: float | None = None) -> dict:
    probabilities = collection.probabilities
    n = collection.num_figures
    mean = expected_boxes(probabilities)

    try:
        spread = std_boxes(probabilities)
    except ValueError:
        spread = float("nan")

    summary = {
        "collection_id": collection.collection_id,
        "collection_name": collection.collection_name,
        "num_figures": n,
        "box_price": collection.box_price,
        "expected_boxes": mean,
        "std_boxes": spread,
        "median_boxes": boxes_for_quantile(probabilities, 0.50),
        "p95_boxes": boxes_for_quantile(probabilities, 0.95),
        "expected_cost": mean * collection.box_price,
        "p95_cost": boxes_for_quantile(probabilities, 0.95) * collection.box_price,
        "uniform_expected_boxes": expected_boxes_uniform(n),
    }

    if budget is not None:
        affordable = int(budget // collection.box_price)
        summary["budget"] = budget
        summary["boxes_affordable"] = affordable
        summary["completion_probability"] = completion_probability(
            probabilities, affordable
        )
    return summary

if __name__ == "__main__":
    import pandas as pd

    rows = [
        analytical_summary(collection, budget=1000.0)
        for collection in load_collections().values()
    ]
    print(pd.DataFrame(rows).to_string(index=False))
