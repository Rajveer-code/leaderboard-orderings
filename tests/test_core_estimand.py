"""Guards the core-estimand correction (CORE_ESTIMAND_AUDIT.md)."""
import sys
import numpy as np

sys.path.insert(0, "scripts")
from fb_core_estimand_audit import g_old, pv_map, pv_interval  # noqa: E402


def _pop(A, alpha, beta):
    a = A * (1 - beta) + (1 - A) * alpha
    q = (1 - A) * alpha / a
    r = A * beta / (1 - a)
    return a, q, r


def test_pv_identity_recovers_true_accuracy():
    for A, al, be in [(0.2, 0.01, 0.02), (0.02, 0.01, 0.0), (0.4, 0.1, 0.15)]:
        a, q, r = _pop(A, al, be)
        assert abs(pv_map(a, q, r) - A) < 1e-12


def test_old_map_fed_predictive_values_is_biased():
    a, q, r = _pop(0.02, 0.01, 0.02)
    assert abs(g_old(a, q, r) - 0.02) > 0.005


def test_pv_interval_never_empty_and_inside_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        a = rng.random()
        qlo, qhi = sorted(rng.random(2)); rlo, rhi = sorted(rng.random(2))
        lo, hi = pv_interval(a, (qlo, qhi, rlo, rhi))
        assert 0 <= lo <= hi <= 1
