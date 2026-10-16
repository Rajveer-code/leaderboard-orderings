"""Real pytest tests for the headline corrected_interval / paired_bootstrap_delta
API -- construct a synthetic DGP with a KNOWN true accuracy, run the public
function, confirm it recovers the truth within its own reported interval a
reasonable fraction of the time (a lightweight version of T3's fuller
coverage simulation, sized to run fast as part of a normal test suite)."""
import numpy as np
import pytest

from flipbudget import corrected_interval, paired_bootstrap_delta
from flipbudget.identification import g


def _simulate(true_a, alpha, beta, n, rng):
    ystar = rng.binomial(1, true_a, size=n)
    yhat = ystar.copy()
    neg, pos = ystar == 0, ystar == 1
    yhat[neg] = rng.binomial(1, alpha, size=neg.sum())
    yhat[pos] = 1 - rng.binomial(1, beta, size=pos.sum())
    return ystar, yhat


def test_corrected_interval_recovers_truth_reasonably_often():
    true_a, alpha, beta = 0.72, 0.03, 0.12
    n_bench, n_audit = 400, 30
    n_trials = 60  # small on purpose -- this is a fast smoke test, not the
    # full T3 coverage simulation (see scripts/fb_t3_coverage.py for that)
    covered = 0
    for i in range(n_trials):
        rng = np.random.default_rng(i)
        _, bench = _simulate(true_a, alpha, beta, n_bench, rng)
        ystar, yhat = _simulate(true_a, alpha, beta, n_audit, rng)
        if (ystar == 0).sum() == 0 or (ystar == 1).sum() == 0:
            continue
        try:
            result = corrected_interval(bench, ystar, yhat, n_boot=300, seed=i)
        except ValueError:
            continue
        if result["interval"] is None:
            continue
        lo, hi = result["interval"]
        if lo <= true_a <= hi:
            covered += 1
    # loose bound: T3's own finding is ~0.90 coverage at this density, not
    # 0.95 -- this smoke test just confirms the function is not wildly broken
    assert covered / n_trials > 0.5


def test_corrected_interval_matches_g_exactly_at_point_estimate():
    rng = np.random.default_rng(0)
    bench = rng.binomial(1, 0.6, size=200)
    ystar, yhat = _simulate(0.6, 0.05, 0.10, 50, rng)
    result = corrected_interval(bench, ystar, yhat, n_boot=100, seed=0)
    expected = g(result["observed_rate"], result["alpha_hat"], result["beta_hat"])
    assert result["corrected_point_estimate"] == pytest.approx(expected)


def test_corrected_interval_rejects_audit_missing_a_class():
    bench = [1, 1, 1, 0, 0]
    ystar = [1, 1, 1, 1, 1]  # no true negatives -- alpha undefined
    yhat = [1, 1, 1, 0, 1]
    with pytest.raises(ValueError):
        corrected_interval(bench, ystar, yhat)


def test_paired_bootstrap_delta_runs_and_returns_expected_keys():
    rng = np.random.default_rng(1)
    b1 = rng.binomial(1, 0.7, size=300)
    b2 = rng.binomial(1, 0.6, size=300)
    ys1, yh1 = _simulate(0.7, 0.03, 0.08, 25, rng)
    ys2, yh2 = _simulate(0.6, 0.02, 0.06, 25, rng)
    result = paired_bootstrap_delta(b1, b2, ys1, yh1, ys2, yh2, n_boot=200, seed=1)
    for key in ["a1", "a2", "corrected_delta_point", "flip_budget_degenerate",
                "coverage_caveat"]:
        assert key in result
