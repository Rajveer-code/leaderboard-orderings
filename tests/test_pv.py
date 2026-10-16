import numpy as np
import pytest

from flipbudget.pv import (accuracy, accuracy_interval, difference_interval,
                           to_error_rates, wilson, sidak_z, indeterminate_counts)


def _pop(A, alpha, beta):
    a = A * (1 - beta) + (1 - A) * alpha
    return a, (1 - A) * alpha / a, A * beta / (1 - a)


@pytest.mark.parametrize("A,al,be", [(0.2, 0.01, 0.02), (0.02, 0.01, 0.0), (0.5, 0.2, 0.1)])
def test_accuracy_identity_and_bayes_roundtrip(A, al, be):
    a, q, r = _pop(A, al, be)
    assert accuracy(a, q, r) == pytest.approx(A)
    assert to_error_rates(a, q, r) == pytest.approx((al, be))


def test_interval_contains_point_and_stays_in_unit_interval():
    rng = np.random.default_rng(1)
    for _ in range(500):
        a = rng.random(); q = sorted(rng.random(2)); r = sorted(rng.random(2))
        lo, hi = accuracy_interval(a, (q[0], q[1], r[0], r[1]))
        assert 0 <= lo <= accuracy(a, q[0], r[0]) <= hi <= 1


def test_difference_interval_is_sharp_at_corners():
    b1, b2 = (0.0, 0.1, 0.0, 0.2), (0.05, 0.3, 0.0, 0.1)
    lo, hi = difference_interval(0.3, b1, 0.2, b2)
    grid = [accuracy(0.3, q1, r1) - accuracy(0.2, q2, r2)
            for q1 in np.linspace(0, .1, 5) for r1 in np.linspace(0, .2, 5)
            for q2 in np.linspace(.05, .3, 5) for r2 in np.linspace(0, .1, 5)]
    assert lo == pytest.approx(min(grid)) and hi == pytest.approx(max(grid))


def test_wilson_zero_events_and_sidak():
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0.2 < hi < 0.35
    assert sidak_z(0.95, 1) == pytest.approx(1.959964, abs=1e-5)
    assert sidak_z(0.95, 4) > sidak_z(0.95, 1)


def test_indeterminate_bounds_order():
    # credited stratum: 1 false credit, 2 indeterminate, 7 fine
    lo, hi = indeterminate_counts(x=1, n=8, k=2)
    assert lo == (1, 8) and hi == (3, 10)
