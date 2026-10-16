"""Full reconciliation: separate sampling / audit-estimation / scorer-identification
/ combined uncertainty for the 136 real pairs, on a CORRECTED, properly-bounded
formulation. Two things being fixed here, both real:

(1) MATH CORRECTION: "accuracy is unbounded" was wrong. A* is a proportion, bounded
    in [0,1] by definition -- always. What actually happens is that the UNCONSTRAINED
    correction expression g(a,alpha,beta)=(a-alpha)/(1-alpha-beta) diverges as
    alpha+beta->1; it does NOT mean the true accuracy becomes undefined or infinite.
    The correct reading: g's value outside [0,1] for some (alpha,beta) in a box means
    THAT (alpha,beta) hypothesis is logically inconsistent with the observed data
    under ANY valid accuracy (the law-of-total-probability identity
    a = A*(1-beta) + (1-A*)*alpha has no solution with A* in [0,1] for that
    (alpha,beta)) -- not that A* itself leaves [0,1]. The identified set for A* is
    therefore {g(a,alpha,beta) : (alpha,beta) in box, denom>0} INTERSECTED WITH
    [0,1], and its width is consequently always <= 1. Implemented directly here
    (bounded_single_model_extrema), replacing the earlier "saturation" bookkeeping
    from fb_tb_compound_design_sensitivity.py, which was mathematically defensible
    as a divergence *diagnostic* but described its own output in the wrong
    vocabulary ("unbounded") for a quantity that cannot exceed [0,1].

(2) THE RECONCILIATION: four uncertainty sources, computed SEPARATELY for the same
    136 real pairs, all using the SAME [0,1]-bounded identification methodology so
    they are directly comparable:
      - sampling            : the field's current textbook practice (E-A's own
                               object, alpha=beta=0 assumed, ignores scorer error).
      - audit-estimation-only: Wilson CI on the audited (alpha,beta), Lambda=1 (no
                               additional sensitivity) -- "how much don't we know
                               because the AUDIT SAMPLE is small."
      - scorer-identification-only: pure Lambda-sensitivity, anchored at the SHRUNK
                               (alpha_pooled,beta_pooled) point estimate (per the
                               TA_BOUNDARY_AUDIT.md fix), NO audit-CI at all --
                               "how much don't we know even if alpha,beta were known
                               exactly, because per-item behavior could differ."
      - combined             : Wilson CI further Lambda-widened (the actual final
                               compound model) -- both sources together.
    Reference Lambda=2.0 for the scorer-only and combined columns, matching the
    "modest sensitivity" convention already used throughout this project.

Run: python scripts/fb_reconcile_layers.py
"""
import json
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "scripts")
from fb_t2_flipbudget import g
from fb_ta_ssm import lambda_bounds
from fb_ta_compound_interval import compound_bounds
from ea_dominance_study import load, build_model_table, wilson_ci

Z = 1.96
L_REFERENCE = 2.0
STRADDLE_CAP = 0.999
GRID_N = 801


def bounded_single_model_extrema(a, alpha_lo, alpha_hi, beta_lo, beta_hi):
    """Identified set for A* = g(a,alpha,beta) over the box, intersected with
    [0,1] -- the correct treatment (see module docstring). Two paths:
    (a) box does not straddle alpha+beta=1 (denom constant-sign throughout):
        exact 4-corner evaluation (T-H vertex theorem, valid here), then clip to
        [0,1].
    (b) box straddles: the true supremum/infimum over the feasible part diverges
        (proven in fb_tb_compound_design_sensitivity.py's saturation_diverges_check
        -- grid max grows without bound as the feasibility cap tightens). A grid
        restricted to the feasible region (alpha+beta<STRADDLE_CAP) at moderate
        resolution ALREADY exceeds [0,1] by a wide margin whenever this happens
        (confirmed: cap=0.999 alone gave values in the hundreds in the earlier
        check) -- so clipping the empirical grid range to [0,1] gives the correct
        answer (the true unconstrained range comfortably contains [0,1] already,
        clipping to it is not an underestimate)."""
    straddles = (alpha_hi + beta_hi >= STRADDLE_CAP) or (alpha_lo + beta_lo >= STRADDLE_CAP)
    if not straddles:
        vals = [g(a, al, be) for al in (alpha_lo, alpha_hi) for be in (beta_lo, beta_hi)]
        lo, hi = min(vals), max(vals)
    else:
        alphas = np.linspace(alpha_lo, alpha_hi, GRID_N)
        betas = np.linspace(beta_lo, beta_hi, GRID_N)
        AA, BB = np.meshgrid(alphas, betas)
        GG = np.where((AA + BB) < STRADDLE_CAP, g(a, AA, BB), np.nan)
        lo, hi = float(np.nanmin(GG)), float(np.nanmax(GG))
    if hi < 0.0 or lo > 1.0:
        # The raw (unclipped) range does not overlap [0,1] AT ALL -- every
        # (alpha,beta) in the box implies an A* outside [0,1], which is
        # impossible (A* is a proportion). This means the assumed (alpha,beta)
        # box is itself inconsistent with the observed a for this model --
        # found auditing this function against real data (17/27 real models'
        # audit-CI boxes hit this; see MATH_COMPARATOR_BUG.md's sibling
        # finding, RECONCILIATION_EMPTY_SET_BUG.md). NOT the same as a narrow-
        # but-valid identified set -- independently clipping min(lo,0) and
        # max(hi,1) here would silently INVERT the interval (lo>hi), which is
        # what this function did before the bug was found. Signal emptiness
        # explicitly rather than fabricate a bounded-looking but nonsensical
        # interval.
        return (float("nan"), float("nan"))
    return max(lo, 0.0), min(hi, 1.0)


def bounded_comparison(a1, box1, a2, box2):
    min1, max1 = bounded_single_model_extrema(a1, *box1)
    min2, max2 = bounded_single_model_extrema(a2, *box2)
    if any(np.isnan(x) for x in (min1, max1, min2, max2)):
        return (float("nan"), float("nan"))
    lo, hi = min1 - max2, max1 - min2
    return max(lo, -1.0), min(hi, 1.0)  # Delta* = A1*-A2*, both in [0,1] -> Delta* in [-1,1]


def sanity_check_nonstraddle_matches_exact():
    """On a box that does NOT straddle, the bounded method must equal plain exact
    corner evaluation with no [0,1] clipping needed (sanity: the clip should be a
    no-op in the ordinary regime)."""
    a, alo, ahi, blo, bhi = 0.6, 0.02, 0.10, 0.05, 0.15
    lo, hi = bounded_single_model_extrema(a, alo, ahi, blo, bhi)
    vals = [g(a, al, be) for al in (alo, ahi) for be in (blo, bhi)]
    exact_lo, exact_hi = min(vals), max(vals)
    ok = abs(lo - exact_lo) < 1e-9 and abs(hi - exact_hi) < 1e-9
    print(f"  non-straddling box: bounded=[{lo:.6f},{hi:.6f}] exact=[{exact_lo:.6f},{exact_hi:.6f}] "
          f"[{'OK' if ok else 'FAIL'}]")
    return ok


def sanity_check_straddle_clips_to_01():
    """On the KNOWN straddling example from the T-B compound audit (x_alpha=2,
    n_alpha=20, x_beta=3, n_beta=15, L=3 -> alpha_hi+beta_hi=1.276, proven to
    diverge), the bounded method must return exactly [0,1] (fully uninformative),
    not a smaller number and not literally unbounded."""
    a = 0.4
    alpha_lo, alpha_hi = compound_bounds(2, 20, 3.0)
    beta_lo, beta_hi = compound_bounds(3, 15, 3.0)
    lo, hi = bounded_single_model_extrema(a, alpha_lo, alpha_hi, beta_lo, beta_hi)
    ok = abs(lo - 0.0) < 1e-6 and abs(hi - 1.0) < 1e-6
    print(f"  known-straddling box: bounded=[{lo:.6f},{hi:.6f}] (expect [0,1]) "
          f"[{'OK' if ok else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    print("=" * 70)
    print("Four-layer reconciliation: sampling | audit-only | scorer-only | combined")
    print("=" * 70)
    print("-" * 70)
    print("Sanity checks on the corrected [0,1]-bounded methodology")
    print("-" * 70)
    ok1 = sanity_check_nonstraddle_matches_exact()
    ok2 = sanity_check_straddle_clips_to_01()
    if not (ok1 and ok2):
        raise SystemExit("Sanity checks failed -- fix before trusting any downstream number.")

    accs, margins = load()
    rows, excl = build_model_table(accs, margins)
    with open("results/flipbudget/e4_mathhard_per_model.json") as f:
        e4 = json.load(f)["per_model"]
    with open("results/analysis/pair_identification_human.json") as f:
        pi = json.load(f)

    print(f"\nModels qualifying: {excl['n_qualifying']} of {excl['n_total_models']}")
    print(f"Reference Lambda for scorer-only and combined columns: {L_REFERENCE}")

    results = []
    n_skipped = 0
    for p in pi["headline"]["pairs"]:
        m_lo, m_hi = p["lo"], p["hi"]
        if m_lo not in rows or m_hi not in rows:
            n_skipped += 1
            continue
        r1, r2 = rows[m_hi], rows[m_lo]
        a1, a2 = r1["a_hat"], r2["a_hat"]
        n_bench1, n_bench2 = r1["n_bench"], r2["n_bench"]

        # 1. sampling (E-A's own object, unchanged)
        w_sampling = 2 * Z * np.sqrt(a1 * (1 - a1) / n_bench1 + a2 * (1 - a2) / n_bench2)
        sampling_contains_zero = abs(a1 - a2) <= w_sampling / 2

        # 2. audit-estimation-only: Wilson CI, Lambda=1, [0,1]-bounded
        box1_audit = wilson_ci(r1["x_alpha"], r1["n0_audit"], Z) + wilson_ci(r1["x_beta"], r1["n1_audit"], Z)
        box2_audit = wilson_ci(r2["x_alpha"], r2["n0_audit"], Z) + wilson_ci(r2["x_beta"], r2["n1_audit"], Z)
        lo_audit, hi_audit = bounded_comparison(a1, box1_audit, a2, box2_audit)
        w_audit = hi_audit - lo_audit
        audit_contains_zero = lo_audit <= 0 <= hi_audit

        # 3. scorer-identification-only: pure Lambda at shrunk anchor, NO audit-CI
        m1, m2 = e4[m_hi], e4[m_lo]
        box1_scorer = lambda_bounds(m1["alpha_pooled"], L_REFERENCE) + lambda_bounds(m1["beta_pooled"], L_REFERENCE)
        box2_scorer = lambda_bounds(m2["alpha_pooled"], L_REFERENCE) + lambda_bounds(m2["beta_pooled"], L_REFERENCE)
        lo_scorer, hi_scorer = bounded_comparison(a1, box1_scorer, a2, box2_scorer)
        w_scorer = hi_scorer - lo_scorer
        scorer_contains_zero = lo_scorer <= 0 <= hi_scorer

        # 4. combined: Wilson CI further Lambda-widened (the actual final model)
        box1_combined = compound_bounds(r1["x_alpha"], r1["n0_audit"], L_REFERENCE) + \
            compound_bounds(r1["x_beta"], r1["n1_audit"], L_REFERENCE)
        box2_combined = compound_bounds(r2["x_alpha"], r2["n0_audit"], L_REFERENCE) + \
            compound_bounds(r2["x_beta"], r2["n1_audit"], L_REFERENCE)
        lo_comb, hi_comb = bounded_comparison(a1, box1_combined, a2, box2_combined)
        w_combined = hi_comb - lo_comb
        combined_contains_zero = lo_comb <= 0 <= hi_comb

        results.append({
            "hi": m_hi, "lo": m_lo, "a1": a1, "a2": a2,
            "w_sampling": w_sampling, "sampling_contains_zero": bool(sampling_contains_zero),
            "w_audit_only": w_audit, "audit_only_contains_zero": bool(audit_contains_zero),
            "w_scorer_only": w_scorer, "scorer_only_contains_zero": bool(scorer_contains_zero),
            "w_combined": w_combined, "combined_contains_zero": bool(combined_contains_zero),
        })

    print(f"\nPairs scored: {len(results)} (skipped {n_skipped})")

    def valid(r, wkey):
        return not np.isnan(r[wkey])

    print()
    print("-" * 70)
    print("RECONCILIATION TABLE (medians across valid pairs only -- see EMPTY-SET NOTE)")
    print("-" * 70)
    print(f"{'layer':<28}{'n valid':<10}{'n excluded':<12}{'median width':<16}"
          f"{'n unresolved':<14}{'%':<8}")
    layer_stats = {}
    for label, wkey, zkey in [
        ("sampling", "w_sampling", "sampling_contains_zero"),
        ("audit-estimation-only", "w_audit_only", "audit_only_contains_zero"),
        (f"scorer-identification-only (L={L_REFERENCE})", "w_scorer_only", "scorer_only_contains_zero"),
        (f"combined (L={L_REFERENCE})", "w_combined", "combined_contains_zero"),
    ]:
        valid_rows = [r for r in results if valid(r, wkey)]
        n_excluded = len(results) - len(valid_rows)
        widths = np.array([r[wkey] for r in valid_rows])
        n_zero = sum(1 for r in valid_rows if r[zkey])
        layer_stats[wkey] = {"n_valid": len(valid_rows), "n_excluded": n_excluded,
                              "median_width": float(np.median(widths)) if len(widths) else None,
                              "n_unresolved": n_zero,
                              "pct_unresolved": 100 * n_zero / len(valid_rows) if valid_rows else None}
        med = f"{np.median(widths):.4f}" if len(widths) else "n/a"
        pct = f"{100*n_zero/len(valid_rows):.1f}%" if valid_rows else "n/a"
        print(f"{label:<28}{len(valid_rows):<10}{n_excluded:<12}{med:<16}{n_zero:<14}{pct:<8}")

    print()
    print("-" * 70)
    print("EMPTY-SET NOTE: why some pairs are excluded per-layer, not silently included")
    print("-" * 70)
    print("A raw (unclipped) g(a,alpha,beta) range that does not overlap [0,1] AT ALL means")
    print("that layer's assumed (alpha,beta) box is itself inconsistent with the model's")
    print("observed accuracy a -- independently clipping min/max to [0,1] in that case would")
    print("silently INVERT the interval (a real bug found and fixed this session, see")
    print("RECONCILIATION_EMPTY_SET_BUG.md). Excluded from that layer's own statistics,")
    print("not assumed resolved or unresolved either way.")

    n_audit_only_zero = layer_stats["w_audit_only"]["n_unresolved"]
    n_scorer_only_zero = layer_stats["w_scorer_only"]["n_unresolved"]
    n_combined_zero = layer_stats["w_combined"]["n_unresolved"]
    n_sampling_zero = layer_stats["w_sampling"]["n_unresolved"]
    n_audit_only_valid = layer_stats["w_audit_only"]["n_valid"]
    n_scorer_only_valid = layer_stats["w_scorer_only"]["n_valid"]
    n_combined_valid = layer_stats["w_combined"]["n_valid"]
    n_sampling_valid = layer_stats["w_sampling"]["n_valid"]

    print()
    print("-" * 70)
    print("DECOMPOSITION: how much of the combined-unresolved count is which layer")
    print("(restricted to pairs valid in BOTH audit-only and scorer-only)")
    print("-" * 70)
    both_valid = [r for r in results if valid(r, "w_audit_only") and valid(r, "w_scorer_only")]
    both = sum(1 for r in both_valid if r["audit_only_contains_zero"] and r["scorer_only_contains_zero"])
    audit_but_not_scorer = sum(1 for r in both_valid if r["audit_only_contains_zero"] and not r["scorer_only_contains_zero"])
    scorer_but_not_audit = sum(1 for r in both_valid if r["scorer_only_contains_zero"] and not r["audit_only_contains_zero"])
    neither = sum(1 for r in both_valid if not r["audit_only_contains_zero"] and not r["scorer_only_contains_zero"])
    print(f"  pairs valid in both layers: {len(both_valid)} of {len(results)}")
    print(f"  unresolved under AUDIT-ONLY but NOT scorer-only: {audit_but_not_scorer} "
          f"(driven purely by finite audit n)")
    print(f"  unresolved under SCORER-ONLY but NOT audit-only: {scorer_but_not_audit} "
          f"(driven purely by Lambda-sensitivity, audit irrelevant)")
    print(f"  unresolved under BOTH: {both}")
    print(f"  resolved under both: {neither}")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  sampling-only unresolved:              {n_sampling_zero}/{n_sampling_valid} valid "
          f"({100*n_sampling_zero/n_sampling_valid:.1f}%)")
    print(f"  audit-estimation-only unresolved:       {n_audit_only_zero}/{n_audit_only_valid} valid "
          f"({100*n_audit_only_zero/n_audit_only_valid:.1f}%)")
    print(f"  scorer-identification-only unresolved:  {n_scorer_only_zero}/{n_scorer_only_valid} valid "
          f"({100*n_scorer_only_zero/n_scorer_only_valid:.1f}%)")
    print(f"  combined unresolved:                    {n_combined_zero}/{n_combined_valid} valid "
          f"({100*n_combined_zero/n_combined_valid:.1f}%)")

    audit_dominant = n_audit_only_zero > 2 * max(n_scorer_only_zero, 1)
    scorer_dominant = n_scorer_only_zero > 2 * max(n_audit_only_zero, 1)
    if scorer_dominant:
        verdict = "A"
        print("\n  A: genuine scorer-induced effect is large -- scorer-only unresolved count")
        print("  substantially exceeds audit-only, so this is not primarily a small-audit artifact.")
    elif audit_dominant:
        verdict = "B"
        print("\n  B: mostly finite-audit uncertainty -- audit-only unresolved count substantially")
        print("  exceeds scorer-only. The combined result is largely a statement about audit size,")
        print("  not about scorer non-uniformity per se.")
    else:
        verdict = "C"
        print("\n  C: mixed, quantified above -- neither layer alone dominates the other by a clear")
        print("  margin; both audit thinness and scorer sensitivity contribute materially.")

    out = {
        "reference_lambda": L_REFERENCE, "n_scored": len(results), "n_skipped": n_skipped,
        "layer_stats": layer_stats,
        "n_sampling_unresolved": n_sampling_zero, "n_sampling_valid": n_sampling_valid,
        "n_audit_only_unresolved": n_audit_only_zero, "n_audit_only_valid": n_audit_only_valid,
        "n_scorer_only_unresolved": n_scorer_only_zero, "n_scorer_only_valid": n_scorer_only_valid,
        "n_combined_unresolved": n_combined_zero, "n_combined_valid": n_combined_valid,
        "decomposition": {"audit_but_not_scorer": audit_but_not_scorer,
                           "scorer_but_not_audit": scorer_but_not_audit,
                           "both": both, "n_valid_in_both": len(both_valid)},
        "verdict": verdict,
        "empty_set_note": ("Pairs where a layer's raw [0,1]-unclipped g() range does not "
            "overlap [0,1] at all are excluded from THAT layer's statistics (see "
            "RECONCILIATION_EMPTY_SET_BUG.md) -- not counted as resolved or unresolved."),
        "per_pair": results,
    }
    # Corrected output, NOT overwriting the original file (preserved as
    # historical record of the pre-fix numbers -- see RECONCILIATION_EMPTY_SET_BUG.md).
    with open("results/flipbudget/reconciliation_four_layers_corrected.json", "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)
    print(f"\nWritten to results/flipbudget/reconciliation_four_layers_corrected.json")
    print("(original results/flipbudget/reconciliation_four_layers.json preserved unchanged)")
    print("=" * 70)
