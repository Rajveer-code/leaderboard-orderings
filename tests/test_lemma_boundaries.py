"""Lemma 1 and the coverage of its interval at the boundaries of Appendix A, against exact
probabilities: heterogeneous sub-populations sampled without replacement, p~ = 0, t = 1 - p~
(where no finite lambda* exists), the lower tail at t = p~, and the case in which no t* exists
(p~^n > alpha/2)."""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flipbudget.certify import kl_interval  # noqa: E402

TOL = 1e-12


def kl(a, b):
    """Bernoulli KL with the conventions of Appendix A."""
    out = 0.0
    for x, y in ((a, b), (1 - a, 1 - b)):
        if x > 0:
            if y <= 0:
                return math.inf
            out += x * math.log(x / y)
    return out


def pmf(pops):
    """Exact distribution of x_c: pops = [(N, K, n)], K credited of N, n drawn without replacement."""
    dist = [1.0]
    for N, K, n in pops:
        h = [math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n) if 0 <= n - k <= N - K else 0.0
             for k in range(n + 1)]
        new = [0.0] * (len(dist) + n)
        for i, a in enumerate(dist):
            for k, b in enumerate(h):
                new[i + k] += a * b
        dist = new
    return dist


def ptilde(pops):
    n = sum(p[2] for p in pops)
    return sum(p[2] / n * p[1] / p[0] for p in pops), n


def random_pops(rng, zero=False, high=False):
    pops = []
    for _ in range(rng.randint(1, 3)):
        N = rng.randint(2, 9)
        K = 0 if zero else (N - rng.randint(0, 1) if high else rng.randint(0, N))
        pops.append((N, K, rng.randint(1, N)))
    return pops


def test_upper_tail_including_t_equal_one_minus_ptilde():
    rng = random.Random(1)
    for _ in range(400):
        pops = random_pops(rng)
        pt, n = ptilde(pops)
        dist = pmf(pops)
        ts = [1 - pt] + [rng.uniform(0, 1 - pt) for _ in range(5)]
        for t in ts:
            if t <= 0:
                continue
            thr = pt + t
            prob = sum(q for k, q in enumerate(dist) if k / n >= thr - 1e-12)
            assert prob <= math.exp(-n * kl(min(thr, 1.0), pt)) + TOL


def test_zero_ptilde_gives_zero_probability():
    rng = random.Random(2)
    for _ in range(100):
        pops = random_pops(rng, zero=True)
        pt, n = ptilde(pops)
        assert pt == 0
        dist = pmf(pops)
        assert sum(dist[1:]) == 0.0
        assert math.exp(-n * kl(0.5, 0.0)) == 0.0


def test_lower_tail_including_t_equal_ptilde():
    rng = random.Random(3)
    for _ in range(400):
        pops = random_pops(rng)
        pt, n = ptilde(pops)
        if pt == 0:
            continue
        dist = pmf(pops)
        for t in [pt] + [rng.uniform(0, pt) for _ in range(5)]:
            thr = pt - t
            prob = sum(q for k, q in enumerate(dist) if k / n <= thr + 1e-12)
            assert prob <= math.exp(-n * kl(max(thr, 0.0), pt)) + TOL


def test_interval_coverage_including_the_case_without_t_star():
    rng = random.Random(4)
    alpha = 0.05
    seen_no_t_star = 0
    for i in range(400):
        pops = random_pops(rng, high=(i % 2 == 0))
        pt, n = ptilde(pops)
        if n * kl(1.0, pt) <= math.log(2 / alpha):
            seen_no_t_star += 1  # p~^n > alpha/2: no t* exists, the upper-tail miss must be impossible
        dist = pmf(pops)
        cover = 0.0
        for k, q in enumerate(dist):
            lo, hi = kl_interval(k, n, 1 - alpha)
            if lo - 1e-9 <= pt <= hi + 1e-9:
                cover += q
        assert cover >= 1 - alpha - TOL
    assert seen_no_t_star > 0  # the boundary case was exercised
