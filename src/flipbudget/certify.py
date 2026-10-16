"""Agreement-stratified ranking certification under bounded heterogeneity H(eps).

Derivations and proofs: THEORY_CERTIFICATION.md. Section numbers below refer to it.
A cell is a verifier-verdict pattern; `conc` maps each concordant cell to the design-valid
interval [L, U] of its pooled correctness, and `disc_*` map each discordant cell to a
per-model interval [l, u] ([0, 1] when the cell is unaudited).
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Callable

from scipy.stats import beta as _beta

Bounds = dict[str, tuple[float, float]]


# ------------------------------------------------------------------ intervals (Lemma 1)
def _kl(p: float, q: float) -> float:
    """Bernoulli KL(p || q)."""
    out = 0.0
    if p > 0:
        out += p * math.log(p / q)
    if p < 1:
        out += (1 - p) * math.log((1 - p) / (1 - q))
    return out


def _bisect(f: Callable[[float], bool], lo: float, hi: float, iters: int = 200) -> float:
    """Boundary between f False (at lo) and f True (at hi)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) else (mid, hi)
    return (lo + hi) / 2


def kl_interval(x: int, n: int, level: float) -> tuple[float, float]:
    """{q : n KL(x/n || q) <= log(2/alpha)}: valid for heterogeneous rates, without replacement."""
    if n == 0:
        return 0.0, 1.0
    p, thr = x / n, math.log(2 / (1 - level)) / n
    lo = 0.0 if x == 0 else _bisect(lambda q: _kl(p, q) <= thr, 0.0, p)
    hi = 1.0 if x == n else _bisect(lambda q: _kl(p, q) > thr, p, 1.0)
    return lo, hi


def cp_interval(x: int, n: int, level: float) -> tuple[float, float]:
    """Clopper-Pearson: exact for identical rates with replacement (reported alongside)."""
    if n == 0:
        return 0.0, 1.0
    a = 1 - level
    lo = 0.0 if x == 0 else float(_beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(_beta.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


def sidak(k: int, level: float = 0.975) -> float:
    """Per-parameter level so that k independent intervals hold jointly at `level`."""
    return level ** (1 / max(k, 1))


def census_bounds(x: int, n: int, size: int) -> tuple[float, float]:
    """Bounds on a finite domain's rate when n of its `size` responses are labelled, x credited.

    A complete census (n == size) knows the rate exactly. Otherwise the unlabelled responses
    could all be credited or all not, which gives the logical range. Nothing is estimated, so
    the bound holds with certainty and adds no parameter to any Sidak allocation
    (TC_EXPANSION_PREREGISTRATION Amendment 6).
    """
    if size <= 0 or not 0 <= x <= n <= size:
        raise ValueError(f"need 0 <= x <= n <= size and size > 0, got x={x}, n={n}, size={size}")
    return x / size, (x + size - n) / size


def audit_param_count(n_conc_cells: int, disc_by_model: dict, pair: tuple | None = None,
                      pooling: bool = True) -> int:
    """Independently estimated audit parameters a certification statement relies on.

    The Sidak allocation must run over exactly these. `pair=(j, k)` gives the parameters one
    ordering uses; `pair=None` gives every parameter on the board, which is what a statement
    about all orderings at once uses. With pooling, each concordant cell contributes one
    interval shared by every model; without it, one interval per model. Each audited
    (model, discordant cell) contributes one per-model interval. `disc_by_model` must list
    every model on the board, with 0 where a model has no audited discordant cell.
    """
    models = tuple(disc_by_model) if pair is None else pair
    conc = n_conc_cells if pooling else n_conc_cells * len(models)
    return conc + sum(disc_by_model.get(m, 0) for m in models)


# ------------------------------------------------------------------ P1: identified range
def lb_cell(a: float, b: float, L: float, U: float, eps: float) -> float:
    """min of a*p_j - b*p_k over R_c(eps) = {p in [lo, hi]^2 : |p_j - p_k| <= eps} (section 3)."""
    lo, hi = max(0.0, L - eps), min(1.0, U + eps)
    if a >= b:
        return a * lo - b * min(hi, lo + eps)
    return a * max(lo, hi - eps) - b * hi


def ub_cell(a: float, b: float, L: float, U: float, eps: float) -> float:
    """max of a*p_j - b*p_k over R_c(eps); the region is symmetric in (p_j, p_k)."""
    return -lb_cell(b, a, L, U, eps)


def pair_range(sj: dict, sk: dict, conc: Bounds, disc_j: Bounds, disc_k: Bounds,
               eps: float) -> tuple[float, float]:
    """[LB, UB] of Delta_jk = A_j - A_k (section 3)."""
    lo = hi = 0.0
    for c, (L, U) in conc.items():
        lo += lb_cell(sj.get(c, 0.0), sk.get(c, 0.0), L, U, eps)
        hi += ub_cell(sj.get(c, 0.0), sk.get(c, 0.0), L, U, eps)
    for c in set(disc_j) | set(disc_k):
        (lj, uj), (lk, uk) = disc_j.get(c, (0.0, 1.0)), disc_k.get(c, (0.0, 1.0))
        a, b = sj.get(c, 0.0), sk.get(c, 0.0)
        lo += a * lj - b * uk
        hi += a * uj - b * lk
    return lo, hi


def model_p_bounds(conc: Bounds, disc: Bounds, eps: float) -> Bounds:
    """Per-model bounds on p_mc implied by the same event (used by the variance bound)."""
    out = {c: (max(0.0, L - eps), min(1.0, U + eps)) for c, (L, U) in conc.items()}
    return out | dict(disc)


# ------------------------------------------------------------------ section 6: sampling
def error_bound(shares: dict, verdict: dict, pb: Bounds) -> float:
    """Upper bound on e^v_m = fraction of model m's rows where Y* != verifier v."""
    return sum(s * ((1 - pb.get(c, (0.0, 1.0))[0]) if verdict[c] else pb.get(c, (0.0, 1.0))[1])
               for c, s in shares.items() if c in verdict)


def halfwidth_paired(disagree: dict, ebar_j: dict, ebar_k: dict, n_items: int, z: float) -> float:
    """z * sqrt(fbar / N), fbar = min over verifiers of d^v + ebar^v_j + ebar^v_k, capped at 1."""
    fbar = min(1.0, min(disagree[v] + ebar_j[v] + ebar_k[v] for v in disagree))
    return z * math.sqrt(fbar / n_items)


def halfwidth_hoeffding(n_items: int, alpha: float) -> float:
    """Distribution-free finite-sample half-width for a mean of D in [-1, 1]."""
    return math.sqrt(2 * math.log(2 / alpha) / n_items)


# ------------------------------------------------------------------ P2: breakdown value
def breakdown_eps(margin: Callable[[float], float], eps_max: float = 1.0, tol: float = 1e-10) -> float:
    """sup{eps : margin(eps) > 0} for a nonincreasing margin = LB(eps) - h(eps)."""
    if margin(0.0) <= 0:
        return 0.0
    if margin(eps_max) > 0:
        return eps_max
    lo, hi = 0.0, eps_max
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if margin(mid) > 0 else (lo, mid)
    return lo


# ------------------------------------------------------------------ family objects
def evaluate(models: list, shares: dict, bounds_for_pair: Callable, halfwidth_for_pair: Callable,
             eps: float) -> dict:
    """Certify every ordering at eps. bounds_for_pair(j, k) -> (conc, disc_j, disc_k);
    halfwidth_for_pair(j, k, eps) -> sampling half-width h (0 for the finite-benchmark estimand)."""
    rows = []
    for j, k in combinations(models, 2):
        conc, dj, dk = bounds_for_pair(j, k)
        lo, hi = pair_range(shares[j], shares[k], conc, dj, dk, eps)
        h = halfwidth_for_pair(j, k, eps)
        winner = j if lo - h > 0 else k if hi + h < 0 else None
        rows.append({"pair": [j, k], "lo": lo, "hi": hi, "h": h,
                     "audit": bool(lo > 0 or hi < 0), "combined": winner is not None, "winner": winner})
    return {"audit_resolved": sum(r["audit"] for r in rows),
            "combined_resolved": sum(r["combined"] for r in rows), "pairs": rows}


def rank_intervals(models: list, certified: set) -> dict:
    """Rank bounds implied by certified (winner, loser) pairs: [1 + #above, M - #below]."""
    M = len(models)
    return {m: (1 + sum(1 for w, l in certified if l == m), M - sum(1 for w, l in certified if w == m))
            for m in models}


# ------------------------------------------------------------------ P4
def n_required(rho: float, power: float) -> int:
    """Smallest n with 1 - (1 - rho)^n >= power (at least one error observed)."""
    n = math.ceil(math.log(1 - power) / math.log(1 - rho))
    while 1 - (1 - rho) ** n < power:
        n += 1
    while n > 1 and 1 - (1 - rho) ** (n - 1) >= power:
        n -= 1
    return n
