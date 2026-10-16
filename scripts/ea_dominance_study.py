"""E-A -- THE DOMINANCE STUDY. Tier 1, item 1. Gates the whole Phase 2 program.

Question: is the identification-width uncertainty (from not knowing the
scorer's true alpha,beta) larger than the sampling-width uncertainty (from
finite benchmark n), on real MATH-Hard data?

Definitions, stated precisely so the comparison is not ad hoc:

  W_sampling(a, n)   = the textbook binomial CI full-width the field
                       currently reports, IGNORING scorer error entirely
                       (i.e. as if alpha=beta=0):
                         2 * 1.96 * sqrt(a*(1-a)/n)

  W_identification(a, alpha_CI, beta_CI)
                     = the width of T1's identified set for A* when alpha
                       and beta are only known to lie in their AUDIT-DERIVED
                       Wilson confidence intervals (NOT an arbitrary +-d
                       perturbation, NOT the shrinkage-pooled point estimate
                       -- the raw, unpooled, per-model Wilson interval, which
                       is the genuine audit-based ignorance about THIS
                       model's true alpha,beta). Computed via T1's proven
                       corner-extrema result.

This directly compares "how much don't we know because the benchmark sample
is finite" against "how much don't we know because the audit sample that
pins down the scorer's error rate is finite" -- the second does NOT shrink
as benchmark n grows, only as audit n grows, and audit n is far smaller
(median 11 per model, vs 1324 benchmark items per model).

Exclusion rule, stated before any result is computed: a model needs
n_used_credited>0 AND n_used_wrong>0 to have a well-defined two-sided Wilson
interval on both alpha and beta. Models failing this are excluded and
counted, not silently dropped.

Both single-model and pairwise (Delta*) versions are computed. Pairwise is
the more decision-relevant one (it is what the flip budget itself operates
on), but single-model is reported too since it needs no per-pair machinery
and is easier to sanity-check independently.

Run: python scripts/ea_dominance_study.py
"""
import json
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "scripts")
from fb_t2_flipbudget import g, single_model_extrema, comparison_extrema

Z = 1.96


def wilson_ci(x_successes: int, n_trials: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval -- better-behaved than normal-approx at small n
    and at p near 0 or 1, both of which occur here (many models have
    alpha_raw=0.0 exactly)."""
    if n_trials == 0:
        return (np.nan, np.nan)
    p = x_successes / n_trials
    denom = 1 + z**2 / n_trials
    center = (p + z**2 / (2 * n_trials)) / denom
    half = z * np.sqrt(p * (1 - p) / n_trials + z**2 / (4 * n_trials**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load():
    with open("results/flipbudget/e1_case_b_TRUE.json") as f:
        e1 = json.load(f)
    with open("results/flipbudget/e4_mathhard_per_model.json") as f:
        e4 = json.load(f)
    return e1["true_accuracy_per_model"], e4["per_model"]


def build_model_table(accs, margins):
    """Per-model: a_hat, n_bench, alpha Wilson CI, beta Wilson CI, plus the
    successes/trials that produced each CI (for sanity-checking)."""
    rows = {}
    n_total = 0
    n_excluded_no_credited = 0
    n_excluded_no_wrong = 0
    for model, acc in accs.items():
        n_total += 1
        if model not in margins:
            continue
        m = margins[model]
        n0, n1 = m["n_used_credited"], m["n_used_wrong"]
        if n0 == 0:
            n_excluded_no_credited += 1
            continue
        if n1 == 0:
            n_excluded_no_wrong += 1
            continue
        alpha_raw = m["alpha_raw"] if m["alpha_raw"] is not None else 0.0
        beta_raw = m["beta_raw"] if m["beta_raw"] is not None else 0.0
        x_alpha = round(alpha_raw * n0)  # false-credit count within credited stratum
        x_beta = round(beta_raw * n1)    # false-miss count within wrong stratum
        alpha_lo, alpha_hi = wilson_ci(x_alpha, n0)
        beta_lo, beta_hi = wilson_ci(x_beta, n1)
        rows[model] = {
            "a_hat": acc["a_hat_true"], "n_bench": acc["n_items_scored"],
            "alpha_raw": alpha_raw, "n0_audit": n0, "x_alpha": x_alpha,
            "alpha_ci": (alpha_lo, alpha_hi),
            "beta_raw": beta_raw, "n1_audit": n1, "x_beta": x_beta,
            "beta_ci": (beta_lo, beta_hi),
        }
    return rows, {
        "n_total_models": n_total,
        "n_qualifying": len(rows),
        "n_excluded_no_credited_audit": n_excluded_no_credited,
        "n_excluded_no_wrong_audit": n_excluded_no_wrong,
    }


def single_model_widths(row):
    a, n = row["a_hat"], row["n_bench"]
    w_sampling = 2 * Z * np.sqrt(a * (1 - a) / n)

    alpha_lo, alpha_hi = row["alpha_ci"]
    beta_lo, beta_hi = row["beta_ci"]
    # T1's box is centered+radius; convert the (possibly asymmetric) Wilson
    # interval into a box by evaluating g at all 4 corners of the ACTUAL
    # [alpha_lo,alpha_hi] x [beta_lo,beta_hi] rectangle (not assuming
    # symmetry around a center, which single_model_extrema assumes -- do
    # this directly here instead of forcing the existing helper's shape).
    vals = []
    for al in (alpha_lo, alpha_hi):
        for be in (beta_lo, beta_hi):
            if al + be >= 0.95:  # matches T3's own precedented denom floor --
                continue          # a corner this close to the alpha+beta->1
                                   # singularity is not a meaningful width,
                                   # it's a near-division-by-zero artifact
            vals.append(g(a, al, be))
    if not vals:
        return w_sampling, None
    w_identification = max(vals) - min(vals)
    return w_sampling, w_identification


def pairwise_widths(row1, row2):
    a1, n1_ = row1["a_hat"], row1["n_bench"]
    a2, n2_ = row2["a_hat"], row2["n_bench"]
    w_sampling = 2 * Z * np.sqrt(a1 * (1 - a1) / n1_ + a2 * (1 - a2) / n2_)

    a1_lo, a1_hi = row1["alpha_ci"]
    b1_lo, b1_hi = row1["beta_ci"]
    a2_lo, a2_hi = row2["alpha_ci"]
    b2_lo, b2_hi = row2["beta_ci"]

    def model_extrema(a, alo, ahi, blo, bhi):
        vals = []
        for al in (alo, ahi):
            for be in (blo, bhi):
                if al + be >= 0.999:
                    continue
                vals.append(g(a, al, be))
        return (min(vals), max(vals)) if vals else (None, None)

    min1, max1 = model_extrema(a1, a1_lo, a1_hi, b1_lo, b1_hi)
    min2, max2 = model_extrema(a2, a2_lo, a2_hi, b2_lo, b2_hi)
    if min1 is None or min2 is None:
        return w_sampling, None
    delta_min = min1 - max2
    delta_max = max1 - min2
    w_identification = delta_max - delta_min
    return w_sampling, w_identification


if __name__ == "__main__":
    print("=" * 70)
    print("E-A -- THE DOMINANCE STUDY (Tier 1, item 1)")
    print("=" * 70)

    accs, margins = load()
    rows, excl = build_model_table(accs, margins)
    print(f"Models: {excl['n_total_models']} total, {excl['n_qualifying']} qualify "
          f"(nonzero audit n on BOTH the credited and wrong strata)")
    print(f"  excluded, no credited-stratum audit item: {excl['n_excluded_no_credited_audit']}")
    print(f"  excluded, no wrong-stratum audit item: {excl['n_excluded_no_wrong_audit']}")
    print()

    # ---- Single-model comparison ----
    print("-" * 70)
    print("SINGLE-MODEL: W_identification vs W_sampling")
    print("-" * 70)
    single_ratios = []
    for model, row in rows.items():
        ws, wi = single_model_widths(row)
        if wi is None:
            continue
        single_ratios.append({"model": model, "w_sampling": ws, "w_identification": wi,
                               "ratio": wi / ws if ws > 0 else np.nan})

    # Filter to finite ratios FIRST, keep the filtered list (not a separate
    # array + separate index into the unfiltered list -- that mismatch was a
    # real bug in an earlier version of this script: it printed the same
    # model as both min AND max. Fixed by indexing one single filtered list
    # throughout, never two different ones for the same lookup.
    finite_rows = [r for r in single_ratios if np.isfinite(r["ratio"])]
    ratios = np.array([r["ratio"] for r in finite_rows])
    print(f"n models scored: {len(ratios)}")
    print(f"  median ratio (W_id / W_sampling): {np.median(ratios):.2f}")
    print(f"  mean ratio: {np.mean(ratios):.2f}")
    print(f"  IQR: [{np.percentile(ratios,25):.2f}, {np.percentile(ratios,75):.2f}]")
    print(f"  fraction with W_identification > W_sampling: "
          f"{(ratios > 1).mean():.3f}")
    imin, imax = int(np.argmin(ratios)), int(np.argmax(ratios))
    print(f"  min ratio: {ratios[imin]:.3f}  (model: {finite_rows[imin]['model']}, "
          f"n0={rows[finite_rows[imin]['model']]['n0_audit']}, "
          f"n1={rows[finite_rows[imin]['model']]['n1_audit']})")
    print(f"  max ratio: {ratios[imax]:.3f}  (model: {finite_rows[imax]['model']}, "
          f"n0={rows[finite_rows[imax]['model']]['n0_audit']}, "
          f"n1={rows[finite_rows[imax]['model']]['n1_audit']})")

    # sanity check: does ratio correlate with audit n (should -- thinner
    # audit -> wider Wilson CI -> larger W_identification, all else equal)
    audit_n = np.array([rows[r["model"]]["n0_audit"] + rows[r["model"]]["n1_audit"]
                         for r in finite_rows])
    corr = np.corrcoef(audit_n, ratios)[0, 1]
    print(f"  sanity check: corr(ratio, total_audit_n) = {corr:.3f} "
          f"(expected negative -- thinner audit, bigger ratio)")

    # Robustness check: restrict to a conventional minimum-cell-count
    # threshold (n0>=5 AND n1>=5, the standard chi-square rule-of-thumb
    # minimum expected-cell-count, not a number chosen to produce a
    # favorable result) and report the SAME statistics restricted to that
    # subset, so thin-audit sensitivity is visible rather than hidden.
    robust_rows = [r for r in finite_rows
                   if rows[r["model"]]["n0_audit"] >= 5 and rows[r["model"]]["n1_audit"] >= 5]
    robust_ratios = np.array([r["ratio"] for r in robust_rows])
    print()
    print(f"  ROBUSTNESS CHECK (n0>=5 AND n1>=5, {len(robust_ratios)} of "
          f"{len(ratios)} models qualify):")
    if len(robust_ratios) > 0:
        print(f"    median ratio: {np.median(robust_ratios):.2f}")
        print(f"    IQR: [{np.percentile(robust_ratios,25):.2f}, "
              f"{np.percentile(robust_ratios,75):.2f}]")
        print(f"    fraction dominant: {(robust_ratios > 1).mean():.3f}")
    else:
        print("    NO models qualify at this threshold -- cannot report a "
              "restricted-sample result. This itself is informative: the real "
              "audit density on this benchmark is thinner than a conventional "
              "minimum-cell-count rule would want.")

    # ---- Pairwise comparison, all real pairs ----
    print()
    print("-" * 70)
    print("PAIRWISE (Delta*): W_identification vs W_sampling, real MATH-Hard pairs")
    print("-" * 70)
    with open("results/analysis/pair_identification_human.json") as f:
        pi = json.load(f)

    pair_ratios = []
    n_pair_skipped = 0
    for p in pi["headline"]["pairs"]:
        m_lo, m_hi = p["lo"], p["hi"]
        if m_lo not in rows or m_hi not in rows:
            n_pair_skipped += 1
            continue
        ws, wi = pairwise_widths(rows[m_hi], rows[m_lo])
        if wi is None or ws == 0:
            n_pair_skipped += 1
            continue
        pair_ratios.append({"lo": m_lo, "hi": m_hi, "w_sampling": ws,
                             "w_identification": wi, "ratio": wi / ws})

    pr = np.array([r["ratio"] for r in pair_ratios])
    print(f"n pairs scored: {len(pr)} (skipped {n_pair_skipped}, missing model data)")
    print(f"  median ratio: {np.median(pr):.2f}")
    print(f"  mean ratio: {np.mean(pr):.2f}")
    print(f"  IQR: [{np.percentile(pr,25):.2f}, {np.percentile(pr,75):.2f}]")
    print(f"  fraction with W_identification > W_sampling: {(pr > 1).mean():.3f}")
    print(f"  fraction with W_identification > 2x W_sampling: {(pr > 2).mean():.3f}")
    print(f"  fraction with W_identification > 10x W_sampling: {(pr > 10).mean():.3f}")

    robust_pairs = [r for r in pair_ratios
                    if rows[r["lo"]]["n0_audit"] >= 5 and rows[r["lo"]]["n1_audit"] >= 5
                    and rows[r["hi"]]["n0_audit"] >= 5 and rows[r["hi"]]["n1_audit"] >= 5]
    robust_pr = np.array([r["ratio"] for r in robust_pairs])
    print()
    print(f"  ROBUSTNESS CHECK (both models n0>=5 AND n1>=5, {len(robust_pr)} of "
          f"{len(pr)} pairs qualify):")
    if len(robust_pr) > 0:
        print(f"    median ratio: {np.median(robust_pr):.2f}")
        print(f"    IQR: [{np.percentile(robust_pr,25):.2f}, "
              f"{np.percentile(robust_pr,75):.2f}]")
        print(f"    fraction dominant: {(robust_pr > 1).mean():.3f}")
    else:
        print("    NO pairs qualify at this threshold.")

    print()
    print("=" * 70)
    print("VERDICT INPUT (not the final verdict -- that also needs T-C, E-E)")
    print("=" * 70)
    dominant_single = (ratios > 1).mean()
    dominant_pair = (pr > 1).mean()
    print(f"Single-model: identification width exceeds sampling width in "
          f"{100*dominant_single:.1f}% of scored models")
    print(f"Pairwise: identification width exceeds sampling width in "
          f"{100*dominant_pair:.1f}% of scored pairs")
    if dominant_single > 0.5 and dominant_pair > 0.5:
        print("RESULT: identification width DOMINATES sampling width in the majority")
        print("of cases on this real data. This does NOT by itself confirm the Phase 2")
        print("headline (T-C and E-E still need to run) but it clears E-A's own bar.")
    else:
        print("RESULT: identification width does NOT dominate in the majority of cases.")
        print("This is evidence AGAINST the headline claim, reported as found, not")
        print("adjusted. See Kill Criterion 1 in MASTERPLAN_PHASE2_FLAGSHIP.md.")
    print("=" * 70)

    out = {
        "exclusions": excl,
        "denom_floor": 0.95,
        "min_audit_n_robustness_threshold": 5,
        "single_model": {
            "n_scored": len(ratios),
            "median_ratio": float(np.median(ratios)),
            "mean_ratio": float(np.mean(ratios)),
            "iqr": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
            "fraction_dominant": float(dominant_single),
            "sanity_corr_ratio_vs_audit_n": float(corr),
            "robust_subset": {
                "n": len(robust_ratios),
                "median_ratio": float(np.median(robust_ratios)) if len(robust_ratios) else None,
                "fraction_dominant": float((robust_ratios > 1).mean()) if len(robust_ratios) else None,
            },
            "per_model": single_ratios,
        },
        "pairwise": {
            "n_scored": len(pr),
            "n_skipped": n_pair_skipped,
            "median_ratio": float(np.median(pr)),
            "mean_ratio": float(np.mean(pr)),
            "iqr": [float(np.percentile(pr, 25)), float(np.percentile(pr, 75))],
            "fraction_dominant": float(dominant_pair),
            "fraction_gt_2x": float((pr > 2).mean()),
            "fraction_gt_10x": float((pr > 10).mean()),
            "robust_subset": {
                "n": len(robust_pr),
                "median_ratio": float(np.median(robust_pr)) if len(robust_pr) else None,
                "fraction_dominant": float((robust_pr > 1).mean()) if len(robust_pr) else None,
            },
            "per_pair": pair_ratios,
        },
    }
    with open("results/flipbudget/ea_dominance_study.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWritten to results/flipbudget/ea_dominance_study.json")
