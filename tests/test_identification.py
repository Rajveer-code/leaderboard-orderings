"""Real pytest tests -- the same assertions verified interactively in
scripts/fb_t1_symbolic.py and scripts/fb_t2_flipbudget.py, packaged as a
runnable regression suite rather than left as one-off scripts only."""
import numpy as np
import pytest

from flipbudget import g, single_model_extrema, comparison_extrema, flip_budget


def test_g_matches_closed_form():
    assert g(0.65, 0.05, 0.10) == pytest.approx((0.65 - 0.05) / (1 - 0.05 - 0.10))


def test_g_can_fall_outside_unit_interval():
    # T1's finding: the identified set is not automatically within [0,1] --
    # that is why callers must intersect with [0,1] themselves if they need it.
    assert g(0.10, 0.15, 0.20) < 0


@pytest.mark.parametrize("a,alpha0,beta0,d", [
    (0.65, 0.075, 0.10, 0.075),
    (0.10, 0.075, 0.10, 0.075),  # non-monotone dg/dalpha regime
    (0.50, 0.245, 0.245, 0.245),  # near the alpha+beta -> 1 boundary
])
def test_single_model_extrema_matches_grid_search(a, alpha0, beta0, d):
    """Grid a box identical to what single_model_extrema itself searches
    (alpha0-d..alpha0+d, beta0-d..beta0+d), intersect with [0,1] (A* is a
    proportion, bounded by definition -- see single_model_extrema's own
    docstring), and confirm the corner-based extrema match to tolerance."""
    n = 401
    alphas = np.linspace(max(0.0, alpha0 - d), min(1.0, alpha0 + d), n)
    betas = np.linspace(max(0.0, beta0 - d), min(1.0, beta0 + d), n)
    AA, BB = np.meshgrid(alphas, betas)
    feasible = (AA + BB) < 0.999
    GG = np.where(feasible, g(a, AA, BB), np.nan)
    grid_min = max(0.0, float(np.nanmin(GG)))
    grid_max = min(1.0, float(np.nanmax(GG)))

    corner_min, corner_max = single_model_extrema(a, alpha0, beta0, d)
    assert corner_min == pytest.approx(grid_min, abs=1e-3)
    assert corner_max == pytest.approx(grid_max, abs=1e-3)


def test_single_model_extrema_always_within_unit_interval_or_empty():
    """A* is a proportion, bounded in [0,1] by definition, always. When the
    box is entirely inconsistent with the data (found on REAL MATH-Hard data
    this project audited, not just a synthetic edge case -- see
    RECONCILIATION_EMPTY_SET_BUG.md: a=0.0 with an audit-derived alpha range
    entirely above 0), the identified set is empty (NaN), not an inverted or
    fabricated bounded interval -- that inversion was the actual bug found
    and fixed here."""
    rng = np.random.default_rng(7)
    n_empty = 0
    for _ in range(500):
        a = rng.uniform(0, 1)
        alpha0, beta0 = rng.uniform(0, 0.4, 2)
        d = rng.uniform(0.01, 0.4)
        lo, hi = single_model_extrema(a, alpha0, beta0, d)
        if np.isnan(lo):
            n_empty += 1
            assert np.isnan(hi)  # both NaN together, never one alone
        else:
            assert 0.0 <= lo <= hi <= 1.0
    assert n_empty > 0  # this random sweep is expected to hit real empty cases


def test_single_model_extrema_empty_set_on_real_case():
    """The exact real case that surfaced this bug: a model with a_hat=0.0
    (CohereForAI/c4ai-command-r-v01 on MATH-Hard) whose audit-derived alpha
    range [0.30, 0.90] is entirely above its own observed accuracy -- every
    (alpha,beta) in the box implies a negative A*, so the identified set is
    empty, not [0, -0.43] (what independent clipping silently produced
    before this fix)."""
    lo, hi = single_model_extrema(a=0.0, alpha0=0.6, beta0=0.09, d=0.3)
    assert np.isnan(lo) and np.isnan(hi)


def test_flip_budget_returns_none_not_fabricated_on_empty_baseline():
    """If the point estimate (alpha0, beta0) alone already implies an
    impossible A* for either model, flip_budget must say so (None,
    degenerate=True), not silently bisect toward a number."""
    d_star, degenerate = flip_budget(a1=0.0, a2=0.5, alpha0=0.6, beta0=0.09)
    assert d_star is None and degenerate is True


def test_single_model_extrema_straddling_box_clips_to_01():
    """The known-straddling example from fb_ta_compound_interval.py's own
    proof (a box whose worst corner crosses alpha+beta=1, where the
    unconstrained supremum/infimum diverges): the correct, bounded answer is
    exactly [0,1] (fully uninformative), not a smaller number and not
    literally unbounded."""
    lo, hi = single_model_extrema(0.4, alpha0=0.5, beta0=0.4, d=0.35)
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(1.0, abs=1e-6)


def test_case_a_preserves_sign():
    rng = np.random.default_rng(42)
    n_fail = 0
    for _ in range(2000):
        a1, a2 = rng.uniform(0, 1, 2)
        alpha, beta = rng.uniform(0, 0.4, 2)
        if alpha + beta >= 0.999:
            continue
        delta = g(a1, alpha, beta) - g(a2, alpha, beta)
        if a1 != a2 and np.sign(delta) != np.sign(a1 - a2):
            n_fail += 1
    assert n_fail == 0


def test_comparison_extrema_collapses_to_point_at_d_zero():
    a1, a2, alpha0, beta0 = 0.70, 0.55, 0.08, 0.12
    mn, mx = comparison_extrema(a1, a2, alpha0, beta0, 0.0)
    expected = g(a1, alpha0, beta0) - g(a2, alpha0, beta0)
    assert mn == pytest.approx(expected, abs=1e-9)
    assert mx == pytest.approx(expected, abs=1e-9)


def test_flip_budget_zero_gap_is_degenerate():
    d_star, degenerate = flip_budget(0.50, 0.50, 0.05, 0.05)
    assert degenerate is True


def test_flip_budget_larger_gap_needs_larger_budget():
    d_small_gap, _ = flip_budget(0.65, 0.62, 0.05, 0.08)
    d_large_gap, _ = flip_budget(0.80, 0.60, 0.02, 0.03)
    assert d_large_gap > d_small_gap
