"""The formulas as stated in the manuscript must equal the tested implementation.

Transcribing a result into prose is a common place for errors to enter, so the paper's
statements are re-implemented here independently from the LaTeX text and checked against
src/flipbudget/certify.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flipbudget import certify as C  # noqa: E402

RNG = np.random.default_rng(20260924)


def paper_min(a, b, L, U, eps):
    """The closed form of Proposition 1 exactly as written in Appendix A (manuscript/tmlr/09_appendix.tex,
    Equation eq:closedform): a*l - b*min(h, l + eps) when a >= b, and a*max(l, h - eps) - b*h
    otherwise, with l = max(0, L - eps) and h = min(1, U + eps)."""
    lo = max(0.0, L - eps)
    hi = min(1.0, U + eps)
    if a >= b:
        return a * lo - b * min(hi, lo + eps)
    return a * max(lo, hi - eps) - b * hi


@pytest.mark.parametrize("seed", range(200))
def test_proposition1_statement_matches_implementation(seed):
    a, b = RNG.uniform(0, 0.4, 2)
    L = RNG.uniform(0, 1)
    U = min(1.0, L + RNG.uniform(0, 0.3))
    eps = RNG.choice([0.0, RNG.uniform(0, 0.3)])
    assert paper_min(a, b, L, U, eps) == pytest.approx(C.lb_cell(a, b, L, U, eps), abs=1e-15)


def test_proposition3_statement():
    """Proposition 2 (prop:oneverifier): Delta_jk = (pi_1 - pi_0)(a_j - a_k) for common predictive values."""
    for _ in range(200):
        pi1, pi0 = RNG.uniform(0, 1, 2)
        a = RNG.uniform(0, 1, 5)
        A = pi0 + a * (pi1 - pi0)
        for j in range(5):
            for k in range(5):
                assert A[j] - A[k] == pytest.approx((pi1 - pi0) * (a[j] - a[k]), abs=1e-12)


def test_eps_zero_collapses_to_the_pooled_analysis():
    """The text claims R_c(0) is the diagonal p_jc = p_kc, i.e. the pooled analysis."""
    for _ in range(100):
        a, b = RNG.uniform(0, 0.4, 2)
        L = RNG.uniform(0, 1)
        U = min(1.0, L + RNG.uniform(0, 0.3))
        common = (a - b) * (L if a >= b else U)  # single shared value at the adverse endpoint
        assert C.lb_cell(a, b, L, U, 0.0) == pytest.approx(common, abs=1e-15)


def test_range_widens_with_eps_so_certification_is_monotone():
    """The text claims certification is monotone in eps because R_c(eps) grows."""
    for _ in range(100):
        a, b = RNG.uniform(0, 0.4, 2)
        L = RNG.uniform(0, 1)
        U = min(1.0, L + RNG.uniform(0, 0.3))
        vals = [C.lb_cell(a, b, L, U, e) for e in np.linspace(0, 0.5, 60)]
        assert all(x >= y - 1e-12 for x, y in zip(vals, vals[1:]))
