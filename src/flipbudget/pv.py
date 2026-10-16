"""Benchmark accuracy from a scorer-output-stratified human audit.

The audit samples within the scorer's output strata, so it estimates
  q = P(Y*=0 | scorer credited)   and   r = P(Y*=1 | scorer marked wrong),
not the error rates alpha = P(credited | Y*=0), beta = P(wrong | Y*=1).
By the law of total probability, true accuracy is affine in (q, r):
  A* = a(1 - q) + (1 - a) r,       a = scorer accuracy (census).
This is the two-phase / verification-bias estimator (Neyman 1938;
Begg & Greenes 1983). See CORE_ESTIMAND_AUDIT.md.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def accuracy(a: float, q: float, r: float) -> float:
    return a * (1 - q) + (1 - a) * r


def to_error_rates(a: float, q: float, r: float) -> tuple[float, float]:
    """Bayes conversion (alpha, beta) from (a, q, r)."""
    A = accuracy(a, q, r)
    alpha = a * q / (1 - A) if A < 1 else 0.0
    beta = (1 - a) * r / A if A > 0 else 0.0
    return alpha, beta


def wilson(x: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = x / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def sidak_z(level: float, k: int) -> float:
    """z for each of k independent intervals so all hold jointly at `level`."""
    return float(norm.ppf(0.5 + level ** (1 / k) / 2))


def accuracy_interval(a: float, box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Sharp range of A* over the rectangle q in [qlo,qhi], r in [rlo,rhi]
    (affine map -> extremes at corners). Never empty, always inside [0,1]."""
    qlo, qhi, rlo, rhi = box
    return accuracy(a, qhi, rlo), accuracy(a, qlo, rhi)


def difference_interval(a1, box1, a2, box2) -> tuple[float, float]:
    """Sharp range of A*_1 - A*_2 when the two models' (q, r) vary
    independently (their audited rows are disjoint)."""
    l1, h1 = accuracy_interval(a1, box1)
    l2, h2 = accuracy_interval(a2, box2)
    return l1 - h2, h1 - l2


def indeterminate_counts(x: int, n: int, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Manski bounds for k indeterminate labels in a stratum with x errors
    among n determinate labels: (errors, trials) with all k counted as
    non-errors, and with all k counted as errors."""
    return (x, n), (x + k, n + k)
