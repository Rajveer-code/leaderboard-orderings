"""T3/T5 -- inference, as a library. The headline API: corrected_interval()
and corrected_comparison() fix someone else's number using their own
benchmark records and their own (small) human-audit sample -- that is the
adoption path this package is built around, not a report-emitting card.

Ported from scripts/fb_t3_coverage.py and scripts/fb_t5_budget_ci.py.
Honest by construction: raw percentile bootstrap is used (not BC, which T3's
own coverage simulation found made things worse, not better -- see that
script's commit history), and every returned interval carries its own
simulated-coverage caveat rather than an unearned "95% CI" label.
"""
from __future__ import annotations

import numpy as np

from .identification import g, flip_budget, DENOM_CAP

DEFAULT_N_BOOT = 1000
COVERAGE_CAVEAT = (
    "This interval is a RAW percentile bootstrap, not a calibrated 95% CI. "
    "flipbudget's own coverage simulation (see the repo's T3 commit history) "
    "found ~0.90-0.92 empirical coverage against a 0.95 nominal target at "
    "realistic audit densities (n=20-50 per model), improving toward nominal "
    "as audit size grows. Report the interval as-is with this caveat, not as "
    "an exact 95% CI -- that would be a stronger claim than the evidence "
    "supports."
)


def _cond_rate(ystar: np.ndarray, yhat: np.ndarray, cond_val: int, target_val: int) -> float:
    mask = ystar == cond_val
    n = mask.sum()
    if n == 0:
        return float("nan")
    return float(((yhat == target_val) & mask).sum() / n)


def corrected_interval(
    benchmark_verdicts,
    audit_ystar,
    audit_yhat,
    n_boot: int = DEFAULT_N_BOOT,
    alpha_level: float = 0.05,
    seed: int = 0,
) -> dict:
    """Correct one model's reported accuracy for scorer misclassification.

    benchmark_verdicts: iterable of 0/1, the scorer's verdict per item on the
        full benchmark run whose reported accuracy you want corrected.
    audit_ystar: iterable of 0/1, a human's judgment of TRUE correctness on a
        (typically much smaller) audited subsample of items.
    audit_yhat: iterable of 0/1, the SAME scorer's verdict on those exact
        audited items (paired with audit_ystar, same order).

    Returns a dict with the naive point estimate, the identification-
    corrected point estimate, a raw percentile bootstrap interval (with an
    honest coverage caveat -- see COVERAGE_CAVEAT), and the point (alpha,beta)
    used.

REQUIREMENT: the audited items must be a simple random sample of benchmark items
(unconditional on the scorer's verdict). If the audit was drawn within the
scorer's credited / not-credited strata, alpha and beta estimated here are
biased; use flipbudget.pv (A* = a(1-q) + (1-a)r) instead. See
CORE_ESTIMAND_AUDIT.md.
"""
    bench = np.asarray(list(benchmark_verdicts), dtype=float)
    ystar = np.asarray(list(audit_ystar), dtype=int)
    yhat = np.asarray(list(audit_yhat), dtype=int)
    if len(ystar) != len(yhat):
        raise ValueError("audit_ystar and audit_yhat must be the same length (paired)")

    a_hat = float(bench.mean())
    alpha_hat = _cond_rate(ystar, yhat, 0, 1)
    beta_hat = _cond_rate(ystar, yhat, 1, 0)
    if np.isnan(alpha_hat) or np.isnan(beta_hat):
        raise ValueError(
            "audit sample has no true-negative or no true-positive items -- "
            "cannot estimate both alpha and beta from it"
        )
    if alpha_hat + beta_hat >= DENOM_CAP:
        raise ValueError(
            f"alpha_hat+beta_hat={alpha_hat+beta_hat:.3f} is too close to 1 -- "
            "the correction is not usable at this margin (see T1's identified-"
            "set derivation for why)"
        )

    corrected_point = g(a_hat, alpha_hat, beta_hat)

    rng = np.random.default_rng(seed)
    n_bench, n_audit = len(bench), len(ystar)
    boot_vals = []
    for _ in range(n_boot):
        a_b = bench[rng.integers(0, n_bench, n_bench)].mean()
        idx = rng.integers(0, n_audit, n_audit)
        ys_b, yh_b = ystar[idx], yhat[idx]
        al_b = _cond_rate(ys_b, yh_b, 0, 1)
        be_b = _cond_rate(ys_b, yh_b, 1, 0)
        if np.isnan(al_b) or np.isnan(be_b) or al_b + be_b >= DENOM_CAP:
            continue
        boot_vals.append(g(a_b, al_b, be_b))

    if len(boot_vals) < n_boot * 0.5:
        interval = None
        n_valid = len(boot_vals)
    else:
        boot_vals = np.array(boot_vals)
        interval = (
            float(np.percentile(boot_vals, 100 * alpha_level / 2)),
            float(np.percentile(boot_vals, 100 * (1 - alpha_level / 2))),
        )
        n_valid = len(boot_vals)

    return {
        "n_bench": n_bench,
        "n_audit": n_audit,
        "observed_rate": a_hat,
        "alpha_hat": alpha_hat,
        "beta_hat": beta_hat,
        "corrected_point_estimate": corrected_point,
        "interval": interval,
        "n_valid_bootstrap_replicates": n_valid,
        "coverage_caveat": COVERAGE_CAVEAT,
    }


def paired_bootstrap_delta(
    bench1, bench2, audit_ystar1, audit_yhat1, audit_ystar2, audit_yhat2,
    n_boot: int = DEFAULT_N_BOOT, alpha_level: float = 0.05, seed: int = 0,
) -> dict:
    """Correct a PAIRWISE comparison (model 1 vs model 2) for scorer
    misclassification, including the flip budget: how much differential
    error (beyond what's already in the two models' own audited margins)
    would it take to put the comparison's sign in doubt.

REQUIREMENT: the audited items must be a simple random sample of benchmark items
(unconditional on the scorer's verdict). If the audit was drawn within the
scorer's credited / not-credited strata, alpha and beta estimated here are
biased; use flipbudget.pv (A* = a(1-q) + (1-a)r) instead. See
CORE_ESTIMAND_AUDIT.md.
"""
    b1 = np.asarray(list(bench1), dtype=float)
    b2 = np.asarray(list(bench2), dtype=float)
    ys1, yh1 = np.asarray(list(audit_ystar1), dtype=int), np.asarray(list(audit_yhat1), dtype=int)
    ys2, yh2 = np.asarray(list(audit_ystar2), dtype=int), np.asarray(list(audit_yhat2), dtype=int)

    a1, a2 = float(b1.mean()), float(b2.mean())
    alpha1, beta1 = _cond_rate(ys1, yh1, 0, 1), _cond_rate(ys1, yh1, 1, 0)
    alpha2, beta2 = _cond_rate(ys2, yh2, 0, 1), _cond_rate(ys2, yh2, 1, 0)
    for name, v in [("alpha1", alpha1), ("beta1", beta1), ("alpha2", alpha2), ("beta2", beta2)]:
        if np.isnan(v):
            raise ValueError(f"{name} is undefined -- one audit sample lacks a "
                              f"needed true-positive or true-negative item")

    alpha0, beta0 = (alpha1 + alpha2) / 2, (beta1 + beta2) / 2
    d_star, degenerate = flip_budget(a1, a2, alpha0, beta0)

    rng = np.random.default_rng(seed)
    n1, n2, na1, na2 = len(b1), len(b2), len(ys1), len(ys2)
    deltas, budgets = [], []
    for _ in range(n_boot):
        a1_b = b1[rng.integers(0, n1, n1)].mean()
        a2_b = b2[rng.integers(0, n2, n2)].mean()
        i1, i2 = rng.integers(0, na1, na1), rng.integers(0, na2, na2)
        al1_b, be1_b = _cond_rate(ys1[i1], yh1[i1], 0, 1), _cond_rate(ys1[i1], yh1[i1], 1, 0)
        al2_b, be2_b = _cond_rate(ys2[i2], yh2[i2], 0, 1), _cond_rate(ys2[i2], yh2[i2], 1, 0)
        if any(np.isnan(v) for v in (al1_b, be1_b, al2_b, be2_b)):
            continue
        if al1_b + be1_b >= DENOM_CAP or al2_b + be2_b >= DENOM_CAP:
            continue
        deltas.append(g(a1_b, al1_b, be1_b) - g(a2_b, al2_b, be2_b))
        a0_b, b0_b = (al1_b + al2_b) / 2, (be1_b + be2_b) / 2
        d_b, deg_b = flip_budget(a1_b, a2_b, a0_b, b0_b)
        budgets.append(0.0 if deg_b else (d_b if d_b is not None else np.nan))

    deltas = np.array([d for d in deltas if np.isfinite(d)])
    budgets = np.array([b for b in budgets if b is not None and np.isfinite(b)])

    return {
        "a1": a1, "a2": a2, "corrected_delta_point": g(a1, alpha1, beta1) - g(a2, alpha2, beta2),
        "delta_interval": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
                            if len(deltas) >= n_boot * 0.5 else None,
        "flip_budget_point": d_star,
        "flip_budget_degenerate": degenerate,
        "flip_budget_lower_5pct": float(np.percentile(budgets, 5)) if len(budgets) >= n_boot * 0.3 else None,
        "coverage_caveat": COVERAGE_CAVEAT,
    }
