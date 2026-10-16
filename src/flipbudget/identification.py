"""T1/T2 -- the identification theory, as a library.

Ported from scripts/fb_t1_symbolic.py and scripts/fb_t2_flipbudget.py, whose
symbolic derivations and Monte-Carlo cross-checks remain the source of truth
for correctness (see tests/test_identification.py, which re-runs the same
assertions as real pytest tests rather than duplicating the derivation).
"""
from __future__ import annotations

import numpy as np

DENOM_CAP = 0.999  # alpha+beta must stay below 1; float safety margin


def g(a: float, alpha: float, beta: float) -> float:
    """The misclassification correction: A* = (a - alpha) / (1 - alpha - beta).

    a: observed (published) rate. alpha: false-credit rate,
    P(scorer says correct | truly wrong). beta: false-miss rate,
    P(scorer says wrong | truly correct)."""
    denom = 1 - alpha - beta
    return (a - alpha) / denom


def single_model_extrema(a: float, alpha0: float, beta0: float, d: float
                          ) -> tuple[float, float]:
    """Identified set for A* = g(a, alpha, beta) over alpha in
    [alpha0-d, alpha0+d], beta in [beta0-d, beta0+d], intersected with [0,1]
    -- A* is a proportion and is bounded in [0,1] BY DEFINITION, always.
    g() itself is unclipped (see test_g_can_fall_outside_unit_interval) and
    can return values outside [0,1] when a given (alpha,beta) is logically
    inconsistent with any valid A* -- that inconsistency means THAT
    (alpha,beta) contributes nothing to the identified set, not that A*
    itself leaves [0,1]. Corrected here to match
    scripts/fb_reconcile_layers.py's bounded_single_model_extrema (the
    package previously left this unclipped -- a real bug in the published
    API, found auditing it against the corrected research methodology).

    Two paths, both already used in the research scripts:
    (a) box does not straddle alpha+beta=1 (denominator has constant sign
        throughout): exact 4-corner evaluation is sharp (T4/T-H's proof),
        then clip to [0,1].
    (b) box straddles: the unconstrained supremum/infimum diverges (proven
        in scripts/fb_tb_compound_design_sensitivity.py); a grid restricted
        to the feasible region already exceeds [0,1] by a wide margin
        whenever this happens, so clipping the grid's range to [0,1] gives
        the correct answer."""
    alpha_lo, alpha_hi = max(0.0, alpha0 - d), min(1.0, alpha0 + d)
    beta_lo, beta_hi = max(0.0, beta0 - d), min(1.0, beta0 + d)
    straddles = (alpha_hi + beta_hi >= DENOM_CAP) or (alpha_lo + beta_lo >= DENOM_CAP)
    if not straddles:
        vals = [g(a, al, be) for al in (alpha_lo, alpha_hi) for be in (beta_lo, beta_hi)]
        lo, hi = min(vals), max(vals)
    else:
        n = 801
        alphas = np.linspace(alpha_lo, alpha_hi, n)
        betas = np.linspace(beta_lo, beta_hi, n)
        AA, BB = np.meshgrid(alphas, betas)
        GG = np.where((AA + BB) < DENOM_CAP, g(a, AA, BB), np.nan)
        lo, hi = float(np.nanmin(GG)), float(np.nanmax(GG))
    if hi < 0.0 or lo > 1.0:
        # Raw range does not overlap [0,1] at all: every (alpha,beta) in this
        # box implies an impossible A*, meaning the box itself is
        # inconsistent with the observed a for this model. Independently
        # clipping lo and hi here would silently INVERT the interval
        # (lo=0.0 > hi<0, or lo>1 > hi=1.0) -- a real bug found auditing this
        # exact function against real MATH-Hard data (see
        # RECONCILIATION_EMPTY_SET_BUG.md in the flipbudget research repo).
        # Signal emptiness explicitly: NaN, not a fabricated bounded interval.
        return (float("nan"), float("nan"))
    return max(lo, 0.0), min(hi, 1.0)


def comparison_extrema(a1: float, a2: float, alpha0: float, beta0: float,
                        d: float) -> tuple[float, float]:
    """Identified interval for Delta* = A*_1 - A*_2 under a shared baseline
    (alpha0, beta0) with each model allowed to deviate independently by up to
    d (T2's Case B). Extrema separate because g1, g2 depend on disjoint free
    variables: max(Delta) = max(g1) - min(g2), min(Delta) = min(g1) - max(g2).
    Both A*_1, A*_2 in [0,1] (single_model_extrema), so Delta* in [-1,1] --
    clipped here to match, mirroring scripts/fb_reconcile_layers.py's
    bounded_comparison. If either model's identified set is empty (NaN --
    see single_model_extrema), the comparison is undefined and NaN
    propagates rather than silently treating a missing model as resolved."""
    min_g1, max_g1 = single_model_extrema(a1, alpha0, beta0, d)
    min_g2, max_g2 = single_model_extrema(a2, alpha0, beta0, d)
    if any(np.isnan(x) for x in (min_g1, max_g1, min_g2, max_g2)):
        return (float("nan"), float("nan"))
    return max(min_g1 - max_g2, -1.0), min(max_g1 - min_g2, 1.0)


def flip_budget(a1: float, a2: float, alpha0: float, beta0: float,
                 d_max: float = 0.5, tol: float = 1e-6
                 ) -> tuple[float | None, bool]:
    """Smallest d such that 0 falls inside the Case-B identified interval for
    Delta* -- the least amount of differential scorer misclassification that
    would put the comparison's sign in doubt. Returns (d_star, degenerate);
    degenerate=True means the comparison is already inconclusive at d=0
    (0 is inside the interval even under non-differential error), so no
    flip-budget number is meaningful -- report that plainly, per
    PREREGISTRATION_FLIPBUDGET.md's degenerate-case rule, rather than a
    fabricated value."""
    min0, max0 = comparison_extrema(a1, a2, alpha0, beta0, 0.0)
    if np.isnan(min0) or np.isnan(max0):
        # The point estimate (alpha0, beta0) alone already implies an
        # impossible A* for a1 or a2 -- the baseline itself is inconsistent
        # with the observed accuracy, a more fundamental problem than
        # "inconclusive at d=0". Reported as such, not silently treated as
        # either degenerate (which implies a valid but uninformative
        # interval) or given a fabricated flip-budget number.
        return None, True
    if min0 <= 0 <= max0:
        return 0.0, True

    minM, maxM = comparison_extrema(a1, a2, alpha0, beta0, d_max)
    if np.isnan(minM) or np.isnan(maxM):
        return None, True
    if not (minM <= 0 <= maxM):
        return None, False  # not flippable within d_max -- report as such, don't fabricate

    lo, hi = 0.0, d_max
    while hi - lo > tol:
        mid = (lo + hi) / 2
        mn, mx = comparison_extrema(a1, a2, alpha0, beta0, mid)
        if mn <= 0 <= mx:
            hi = mid
        else:
            lo = mid
    return hi, False
