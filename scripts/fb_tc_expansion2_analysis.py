"""Item 5 of the latest directive: the complete analysis pipeline for the
preregistered T-C expansion (TC_EXPANSION_PREREGISTRATION.md), written and
tested BEFORE valid labels exist, so a genuine submission runs through this
unchanged the moment it arrives.

Steps (fixed by the preregistration, not decided here):
  0. QUALITY GATE -- the same statistical-signature check that caught the
     two excluded submissions (TC_EXPANSION2_LABELING_QUALITY_ISSUE.md),
     now automated. Refuses to proceed silently; requires an explicit
     --i-confirm-human-labeled flag to override, so a human consciously
     signs off rather than the script making that call on its own.
  1. Recompute alpha (both_credited cell) and beta (both_not_credited
     cell) for exact_match AND score_boxed separately.
  2. Recompute the disagreement-resolution rate from the two disagreement
     cells (does the human tend to agree with exact_match or score_boxed
     when they disagree).
  3. Recompute scorer-identification / audit-estimation / combined
     uncertainty for BOTH scorer targets using the existing, unmodified
     fb_reconcile_layers.py machinery.
  4. Recompute pairwise identification, compare directly against the
     historical 69-row (score_boxed-only) result.
  5. Report the A/B/C/D interpretation strictly from the numbers -- not
     decided in advance.

Run (self-test on synthetic placeholder data):
  python scripts/fb_tc_expansion2_analysis.py --self-test
Run (real submission, after manual review of the quality-gate output):
  python scripts/fb_tc_expansion2_analysis.py --csv labeling/tc_expansion2_l1.csv --i-confirm-human-labeled
"""
import argparse
import csv
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from ea_dominance_study import wilson_ci
from fb_reconcile_layers import bounded_single_model_extrema, bounded_comparison

Z = 1.96


def load_provenance():
    with open("results/analysis/tc_expansion2_provenance.json", encoding="utf-8") as f:
        return {r["uid"]: r for r in json.load(f)["rows"]}


def load_mathutils():
    import importlib.util
    import types
    if "datasets" not in sys.modules:
        stub = types.ModuleType("datasets")
        stub.Dataset = object
        sys.modules["datasets"] = stub
    spec = importlib.util.spec_from_file_location("mathutils", "vendor/leaderboard_math/utils.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def matches_gold(mathutils, answer, gold):
    from fb_math_comparator_fixed import is_equiv_fixed
    if answer in ("NONE", "CONTRADICTORY", ""):
        return None
    try:
        a = mathutils.normalize_final_answer(answer)
        g = mathutils.normalize_final_answer(gold)
    except Exception:
        return None
    return a.strip() == g.strip() or is_equiv_fixed(a, g)


def load_labeled_csv(path):
    """Reads the CSV preserving duplicate-uid evidence (a plain dict
    comprehension over DictReader silently collapses duplicate uids to
    their last value, destroying the evidence a quality gate needs to see).
    Returns (labeled dict, duplicate_uids list, n_rows_in_file)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    seen, dupes = set(), []
    labeled = {}
    for r in rows:
        uid = r["uid"]
        if uid in seen:
            dupes.append(uid)
        seen.add(uid)
        labeled[uid] = r["your_answer"]
    return labeled, dupes, len(rows)


def quality_gate(labeled, prov, mathutils, n_rows_in_file=None, duplicate_uids=None):
    """Returns (passed, report_dict). Mirrors the checks that caught the
    two excluded submissions: near-total agreement with both automated
    scorers in the agreement cells, near-zero match in the disagreement-
    with-both-scorers cell, and zero CONTRADICTORY usage -- plus structural
    checks (duplicate/unmatched uids, expected row count, answer-text
    corruption heuristics) added after those same two incidents, to catch a
    THIRD bad submission before it ever reaches the statistical checks."""
    by_cell = {}
    n_unmatched = 0
    corrupted = []
    for uid, ans in labeled.items():
        p = prov.get(uid)
        if not p:
            n_unmatched += 1
            continue
        m = matches_gold(mathutils, ans, p["gold"])
        by_cell.setdefault(p["cell"], []).append(m)
        stripped = ans.strip()
        if stripped not in ("NONE", "CONTRADICTORY") and (
                "\n" in ans or "\r" in ans or len(stripped) > 200):
            corrupted.append(uid)

    n_contra = sum(1 for v in labeled.values() if v.strip() == "CONTRADICTORY")
    n_none = sum(1 for v in labeled.values() if v.strip() == "NONE")

    cell_rates = {}
    for cell, ms in by_cell.items():
        n = len(ms)
        n_match = sum(1 for m in ms if m is True)
        n_scored = sum(1 for m in ms if m is not None)
        cell_rates[cell] = {"n": n, "n_match": n_match, "n_scored": n_scored,
                             "match_rate": n_match / n_scored if n_scored else None}

    red_flags = []
    if n_contra == 0:
        red_flags.append("zero CONTRADICTORY used across the whole sheet")
    bc = cell_rates.get("both_credited", {})
    if bc.get("match_rate") is not None and bc["match_rate"] > 0.98:
        red_flags.append(f"both_credited match rate implausibly high ({bc['match_rate']:.1%})")
    bnc = cell_rates.get("both_not_credited", {})
    if bnc.get("match_rate") is not None and bnc["match_rate"] < 0.02:
        red_flags.append(f"both_not_credited match rate implausibly low ({bnc['match_rate']:.1%}) "
                          f"-- exactly reproduces the historical near-zero result on FRESH data")
    if duplicate_uids:
        red_flags.append(f"{len(duplicate_uids)} duplicate uid(s) in the submitted CSV "
                          f"(each collapses to its LAST occurrence unless fixed first)")
    if n_unmatched:
        red_flags.append(f"{n_unmatched} uid(s) in the submission do not match any row in "
                          f"tc_expansion2_provenance.json -- wrong file, or hand-edited uids")
    if n_rows_in_file is not None and n_rows_in_file != 458:
        red_flags.append(f"{n_rows_in_file} rows in the file, expected 458 (the preregistered "
                          f"200/198/30/30 design) -- a partial or merged submission")
    if corrupted:
        red_flags.append(f"{len(corrupted)} answer(s) contain embedded newlines or exceed 200 "
                          f"chars -- resembles the corruption pattern in the second excluded "
                          f"submission (trailing junk text from a bulk edit), not a normal answer")

    passed = len(red_flags) < 2  # one borderline flag alone doesn't block; two or more does
    return passed, {"cell_rates": cell_rates, "n_contradictory": n_contra, "n_none": n_none,
                     "n_unmatched_uids": n_unmatched, "n_duplicate_uids": len(duplicate_uids or []),
                     "n_corrupted_answers": len(corrupted), "corrupted_uids": corrupted[:20],
                     "red_flags": red_flags}


# Population sizes per preregistered cell (eligible, non-overlapping with the
# existing 400-item audit) -- fixed by the design, not re-derived from the
# sample. Cross-checked directly against tc_expansion_population.json.
CELL_POPULATION = {"both_credited": 1531, "both_not_credited": 19873,
                    "disagree_sb_credited_em_not": 291, "disagree_em_credited_sb_not": 493}

# Which cells make up each scorer's credited/wrong union -- NOT the same two
# cells for both scorers (item 5 of the audit): score_boxed's "credited"
# stratum is both_credited + disagree_sb_credited_em_not; exact_match's is
# both_credited + disagree_em_credited_sb_not. Verified against the cell
# definitions in fb_tc_expansion_sample.py's cell_of().
SCORER_CELLS = {
    "score_boxed_stratum": {"credited": ["both_credited", "disagree_sb_credited_em_not"],
                             "wrong": ["both_not_credited", "disagree_em_credited_sb_not"]},
    "exact_match_stratum": {"credited": ["both_credited", "disagree_em_credited_sb_not"],
                             "wrong": ["both_not_credited", "disagree_sb_credited_em_not"]},
}


def _ht_pool(cell_counts, cells):
    """Horvitz-Thompson / post-stratified pooling across cells with
    DIFFERENT sampling fractions relative to their population sizes --
    NOT a naive pooled proportion. Naive pooling here is a real bias risk:
    verified numerically (this audit's own methodological review) that if
    a small, densely-oversampled cell (e.g. disagree_em_credited_sb_not,
    sampled at ~6x the rate of both_not_credited) has a true rate that
    differs from the majority cell -- exactly the scenario stratifying on
    disagreement was designed to detect -- naive pooling can be biased by
    several hundred percent relative to the population-weighted truth.
    Returns (point_estimate, standard_error) with finite-population
    correction; falls back gracefully when a cell has zero scored rows."""
    num, den, var = 0.0, 0.0, 0.0
    detail = {}
    for c in cells:
        n_c, x_c = cell_counts.get(c, (0, 0))
        Nh = CELL_POPULATION[c]
        if n_c == 0:
            detail[c] = {"n": 0, "x": 0, "p": None, "N_pop": Nh}
            continue
        ph = x_c / n_c
        num += Nh * ph
        den += Nh
        fpc = max(1 - n_c / Nh, 0.0)
        var += (Nh ** 2) * fpc * ph * (1 - ph) / max(n_c - 1, 1)
        detail[c] = {"n": n_c, "x": x_c, "p": ph, "N_pop": Nh}
    if den == 0:
        return None, None, detail
    p_hat = num / den
    se = np.sqrt(var) / den
    return p_hat, se, detail


def recompute_alpha_beta(labeled, prov, mathutils, scorer_field):
    """scorer_field: 'score_boxed_stratum' or 'exact_match_stratum'. Uses
    the design-weighted (Horvitz-Thompson) estimator across the cells that
    make up this scorer's credited/wrong union -- see _ht_pool docstring."""
    cell_counts_credit_wrong = {}  # cell -> (n_credited_scored, n_false_credit), (n_wrong_scored, n_false_miss)
    per_cell_n = {}
    per_cell_false_credit = {}
    per_cell_n_wrong = {}
    per_cell_false_miss = {}
    for uid, ans in labeled.items():
        p = prov.get(uid)
        if not p:
            continue
        m = matches_gold(mathutils, ans, p["gold"])
        if m is None:
            continue
        cell = p["cell"]
        is_credited = p[scorer_field] == "credited"
        if is_credited:
            per_cell_n[cell] = per_cell_n.get(cell, 0) + 1
            if not m:
                per_cell_false_credit[cell] = per_cell_false_credit.get(cell, 0) + 1
        else:
            per_cell_n_wrong[cell] = per_cell_n_wrong.get(cell, 0) + 1
            if m:
                per_cell_false_miss[cell] = per_cell_false_miss.get(cell, 0) + 1

    credited_cells = SCORER_CELLS[scorer_field]["credited"]
    wrong_cells = SCORER_CELLS[scorer_field]["wrong"]
    alpha_counts = {c: (per_cell_n.get(c, 0), per_cell_false_credit.get(c, 0)) for c in credited_cells}
    beta_counts = {c: (per_cell_n_wrong.get(c, 0), per_cell_false_miss.get(c, 0)) for c in wrong_cells}
    alpha, alpha_se, alpha_detail = _ht_pool(alpha_counts, credited_cells)
    beta, beta_se, beta_detail = _ht_pool(beta_counts, wrong_cells)

    n_credited_total = sum(per_cell_n.get(c, 0) for c in credited_cells)
    n_fc_total = sum(per_cell_false_credit.get(c, 0) for c in credited_cells)
    n_wrong_total = sum(per_cell_n_wrong.get(c, 0) for c in wrong_cells)
    n_fm_total = sum(per_cell_false_miss.get(c, 0) for c in wrong_cells)
    return {"n_credited": n_credited_total, "n_false_credit": n_fc_total,
            "alpha": alpha, "alpha_se": alpha_se, "alpha_cell_detail": alpha_detail,
            "n_wrong": n_wrong_total, "n_false_miss": n_fm_total,
            "beta": beta, "beta_se": beta_se, "beta_cell_detail": beta_detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="labeling/tc_expansion2_l1.csv")
    ap.add_argument("--i-confirm-human-labeled", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                     help="Run on synthetic placeholder data to verify the pipeline, not real labels.")
    args = ap.parse_args()

    mathutils = load_mathutils()
    prov = load_provenance()

    if args.self_test:
        print("=" * 78)
        print("SELF-TEST MODE -- synthetic placeholder labels, NOT real data, NOT a result")
        print("=" * 78)
        rng = np.random.default_rng(0)
        labeled = {}
        for uid, p in prov.items():
            # Synthetic "human": agrees with gold ~85% in agreement cells, ~15% in the
            # both-not-credited cell (deliberately NOT near-zero, to prove the pipeline
            # doesn't just reproduce whatever pattern is fed to it) -- pure smoke test.
            if rng.random() < (0.85 if p["cell"] != "both_not_credited" else 0.15):
                labeled[uid] = p["gold"]
            else:
                labeled[uid] = "999999"  # deliberately wrong placeholder
        # inject some CONTRADICTORY/NONE so the smoke test isn't itself flagged
        keys = list(labeled.keys())
        for k in keys[:20]:
            labeled[k] = "CONTRADICTORY"
        for k in keys[20:60]:
            labeled[k] = "NONE"
        n_rows_in_file, dupes = None, []
    else:
        labeled, dupes, n_rows_in_file = load_labeled_csv(args.csv)

    passed, gate_report = quality_gate(labeled, prov, mathutils,
                                        n_rows_in_file=n_rows_in_file, duplicate_uids=dupes)
    print("-" * 78)
    print("QUALITY GATE")
    print("-" * 78)
    print(json.dumps(gate_report, indent=2))
    if not passed and not args.i_confirm_human_labeled and not args.self_test:
        print("\nGATE FAILED -- statistical signature resembles the two excluded submissions.")
        print("Re-run with --i-confirm-human-labeled only after manually confirming this")
        print("was produced by independent human reading, not an automated tool.")
        sys.exit(1)
    elif not passed:
        print("\n[gate would have failed -- proceeding because self-test or override was given]")

    print("\n" + "=" * 78)
    print("STEP 1: alpha/beta, both scorers")
    print("=" * 78)
    sb = recompute_alpha_beta(labeled, prov, mathutils, "score_boxed_stratum")
    em = recompute_alpha_beta(labeled, prov, mathutils, "exact_match_stratum")
    print(f"score_boxed: alpha={sb['alpha']} ({sb['n_false_credit']}/{sb['n_credited']})  "
          f"beta={sb['beta']} ({sb['n_false_miss']}/{sb['n_wrong']})")
    print(f"exact_match: alpha={em['alpha']} ({em['n_false_credit']}/{em['n_credited']})  "
          f"beta={em['beta']} ({em['n_false_miss']}/{em['n_wrong']})")

    print("\n" + "=" * 78)
    print("STEP 2: disagreement-resolution rate")
    print("=" * 78)
    disagree_cells = ["disagree_sb_credited_em_not", "disagree_em_credited_sb_not"]
    for cell in disagree_cells:
        agree_with_sb, agree_with_em, n_scored = 0, 0, 0
        for uid, ans in labeled.items():
            p = prov.get(uid)
            if not p or p["cell"] != cell:
                continue
            m = matches_gold(mathutils, ans, p["gold"])
            if m is None:
                continue
            n_scored += 1
            sb_credited = p["score_boxed_stratum"] == "credited"
            if m == sb_credited:
                agree_with_sb += 1
            else:
                agree_with_em += 1
        print(f"  {cell}: n_scored={n_scored} human_agrees_with_score_boxed={agree_with_sb} "
              f"human_agrees_with_exact_match={agree_with_em}")

    print("\n" + "=" * 78)
    print("STEPS 3-4: reconciliation layers + pairwise comparison, both scorer targets")
    print("=" * 78)
    from ea_dominance_study import load as load_ea, build_model_table as build_ea
    from fb_reconcile_layers import L_REFERENCE
    from fb_ta_ssm import lambda_bounds
    with open("results/flipbudget/e4_mathhard_per_model.json") as f:
        e4 = json.load(f)["per_model"]
    with open("results/analysis/pair_identification_human.json") as f:
        pair_pop = json.load(f)["headline"]["pairs"]
    accs, margins = load_ea()
    rows, _ = build_ea(accs, margins)

    recon_out = {}
    for label, ab in (("score_boxed", sb), ("exact_match", em)):
        if ab["alpha"] is None or ab["beta"] is None or ab["n_credited"] == 0 or ab["n_wrong"] == 0:
            print(f"  {label}: insufficient scored rows for a pooled alpha/beta -- skipped")
            continue
        # Design-based CI (normal approx around the HT point estimate using its
        # design SE) -- NOT a Wilson CI on the naively pooled raw counts, which
        # would assume simple random sampling within each scorer's credited/
        # wrong union. This audit's disproportionate cell sampling fractions
        # (e.g. both_not_credited at ~1% vs disagree_em_credited_sb_not at
        # ~6%) make that assumption wrong -- verified numerically to bias a
        # naive estimate by several hundred percent when the small, densely-
        # sampled cell's true rate differs from the majority cell's.
        alpha_lo = max(0.0, ab["alpha"] - Z * ab["alpha_se"])
        alpha_hi = min(1.0, ab["alpha"] + Z * ab["alpha_se"])
        beta_lo = max(0.0, ab["beta"] - Z * ab["beta_se"])
        beta_hi = min(1.0, ab["beta"] + Z * ab["beta_se"])
        n_unresolved, n_valid_pairs, widths = 0, 0, []
        for p in pair_pop:
            lo_m, hi_m = p["lo"], p["hi"]
            if lo_m not in rows or hi_m not in rows:
                continue
            r1, r2 = rows[hi_m], rows[lo_m]
            box = (alpha_lo, alpha_hi, beta_lo, beta_hi)
            lo_c, hi_c = bounded_comparison(r1["a_hat"], box, r2["a_hat"], box)
            if np.isnan(lo_c):
                continue
            n_valid_pairs += 1
            widths.append(hi_c - lo_c)
            if lo_c <= 0 <= hi_c:
                n_unresolved += 1
        med = float(np.median(widths)) if widths else None
        print(f"  {label}: pooled alpha=[{alpha_lo:.4f},{alpha_hi:.4f}] beta=[{beta_lo:.4f},{beta_hi:.4f}]  "
              f"n_valid_pairs={n_valid_pairs}  unresolved={n_unresolved}  median_width={med}")
        recon_out[label] = {"alpha_ci": [alpha_lo, alpha_hi], "beta_ci": [beta_lo, beta_hi],
                             "n_valid_pairs": n_valid_pairs, "n_unresolved": n_unresolved,
                             "median_width": med}

    print(f"\n  Historical 69-row (score_boxed) comparison: see TC_REAL_ANALYSIS.md (0/69 scorer-wrong)")

    print("\n" + "=" * 78)
    print("ANALYSIS C: the exact_match-self-consistent dominance ratio")
    print("=" * 78)
    print("Mirrors ea_dominance_canonical.py's methodology exactly (same sampling-width")
    print("formula, same 'dominant' = identification width > sampling width definition),")
    print("using exact_match's OWN accuracy (already this project's a_hat throughout)")
    print("paired with exact_match's OWN newly-audited box -- the fully self-consistent")
    print("analysis A/B/C table needs, not previously computed anywhere in this pipeline.")
    analysis_c_ratio = None
    em_box = recon_out.get("exact_match")
    if em_box is not None:
        box = tuple(em_box["alpha_ci"]) + tuple(em_box["beta_ci"])
        ratios, n_valid_c, n_dominant_c = [], 0, 0
        for p in pair_pop:
            lo_m, hi_m = p["lo"], p["hi"]
            if lo_m not in rows or hi_m not in rows:
                continue
            r1, r2 = rows[hi_m], rows[lo_m]
            a1, a2 = r1["a_hat"], r2["a_hat"]
            w_sampling = 2 * Z * np.sqrt(a1 * (1 - a1) / r1["n_bench"] + a2 * (1 - a2) / r2["n_bench"])
            lo_c, hi_c = bounded_comparison(a1, box, a2, box)
            if np.isnan(lo_c) or w_sampling <= 0:
                continue
            n_valid_c += 1
            ratio = (hi_c - lo_c) / w_sampling
            ratios.append(ratio)
            if ratio > 1:
                n_dominant_c += 1
        if ratios:
            analysis_c_ratio = {
                "ratio_median": float(np.median(ratios)), "ratio_mean": float(np.mean(ratios)),
                "ratio_min": float(np.min(ratios)), "ratio_max": float(np.max(ratios)),
                "n_valid": n_valid_c, "n_dominant": n_dominant_c,
                "pct_dominant": 100 * n_dominant_c / n_valid_c,
            }
            print(f"  median ratio={analysis_c_ratio['ratio_median']:.4f}  "
                  f"pct_dominant={analysis_c_ratio['pct_dominant']:.1f}%  n_valid={n_valid_c}")
        else:
            print("  no valid pairs under exact_match's newly-audited box (empty identified sets "
                  "or all pairs excluded) -- analysis C genuinely undetermined, not silently zero")
    else:
        print("  exact_match alpha/beta insufficiently scored -- analysis C not computed")

    print("\n" + "=" * 78)
    print("STEP 5: A/B/C/D interpretation -- NOT decided here, computed from the numbers above")
    print("=" * 78)
    print("A (scorer-error-dominant): beta or alpha for either scorer turns out large")
    print("B (audit-uncertainty-dominant): both alpha/beta stay small, audit-only width still >> scorer-only")
    print("C (both matter): mixed signal across the two scorers/layers")
    print("D (small-audit artifact): the 5.6x perfect-scorer-at-n=11 phenomenon (NEGATIVE_CONTROLS.md)")
    print("   resolves toward stability once this audit's added power is folded in")
    print("Read directly off steps 1-4's real numbers once they exist -- no placeholder verdict given here.")

    out = {"quality_gate": gate_report, "gate_passed": passed, "score_boxed_alpha_beta": sb,
           "exact_match_alpha_beta": em, "reconciliation_by_scorer": recon_out,
           "analysis_c_ratio": analysis_c_ratio, "self_test": args.self_test}
    suffix = "_selftest" if args.self_test else ""
    with open(f"results/flipbudget/tc_expansion2_analysis{suffix}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to results/flipbudget/tc_expansion2_analysis{suffix}.json")


if __name__ == "__main__":
    main()
