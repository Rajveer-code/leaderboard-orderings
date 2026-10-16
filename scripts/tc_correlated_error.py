"""T-C -- does correlated/heterogeneous scorer error materially CANCEL the
differential component? Tier 1, item 2. This is the objection that would
sink the paper if unanswered, and the theoretical prediction (Phase 2 plan,
T-C) is that a SHARED per-item "scorer difficulty" factor would make the
naive independent-error bound conservative -- i.e. correlated error would
make things LOOK worse than they are, not worse than they look.

Test: on items audited for >=2 different models (46 of 350 distinct items
in the real MATH-Hard audit -- a genuinely thin subsample, reported as
such, not treated as large-n), is scorer misclassification (Yhat != Y*, per
model per item) correlated ACROSS models on the SAME item, beyond what each
model's own independent marginal wrongness rate would predict?

If yes (positive correlation): a shared item-difficulty factor exists;
T-C's cancellation argument has empirical support; the naive independent
bound is conservative.
If no (no correlation, or negative): misclassification is closer to
independent across models; T-C's cancellation argument does NOT apply here;
the naive bound is not obviously conservative for this reason (though it
may still be for others).

Explicitly: this script does NOT assume the answer. It reports whichever
result the real data gives.

Run: python scripts/tc_correlated_error.py
"""
import csv
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

if "datasets" not in sys.modules:
    _stub = types.ModuleType("datasets")
    _stub.Dataset = object
    sys.modules["datasets"] = _stub

_spec = importlib.util.spec_from_file_location(
    "leaderboard_math_utils", Path("vendor/leaderboard_math/utils.py"))
mathutils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mathutils)

if not hasattr(__import__("signal"), "SIGALRM"):
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout
    _orig = mathutils.is_equiv
    _pool = ThreadPoolExecutor(max_workers=4)

    def _is_equiv_deadline(x1, x2, seconds=1.0):
        fut = _pool.submit(_orig, x1, x2)
        try:
            return fut.result(timeout=seconds)
        except FutureTimeout:
            return False
    mathutils.is_equiv = _is_equiv_deadline


def is_indeterminate(answer):
    if answer is None or answer.strip() == "":
        return True
    return answer.strip().upper() in ("NONE", "CONTRADICTORY")


def matches_gold(answer, gold):
    a = mathutils.normalize_final_answer(answer)
    g = mathutils.normalize_final_answer(gold)
    return a.strip() == g.strip() or mathutils.is_equiv(a, g)


def load():
    with open("results/analysis/mathhard_labelling_key.json") as f:
        key = json.load(f)
    with open("results/analysis/mathhard_l1.csv", newline="", encoding="utf-8") as f:
        answers = {row["uid"]: row["your_answer"] for row in csv.DictReader(f)}
    return key, answers


def scorer_wrong(item, answers):
    """Yhat != Y*, i.e. the scorer's own verdict (implied by stratum)
    disagrees with the human's determination of correctness. Returns None
    for indeterminate/excluded items (matches E4's own exclusion policy)."""
    ans = answers.get(item["uid"])
    if is_indeterminate(ans):
        return None
    human_says_correct = matches_gold(ans, item["gold"])
    scorer_says_correct = (item["stratum"] == "credited")
    return scorer_says_correct != human_says_correct  # True = scorer wrong


if __name__ == "__main__":
    print("=" * 70)
    print("T-C -- correlated/heterogeneous scorer error, Tier 1 item 2")
    print("=" * 70)
    key, answers = load()

    by_item = defaultdict(list)
    for item in key:
        by_item[item["item_id"]].append(item)
    shared_items = {k: v for k, v in by_item.items() if len(v) >= 2}
    print(f"Distinct audited items: {len(by_item)}")
    print(f"Items shared across >=2 models: {len(shared_items)} "
          f"({100*len(shared_items)/len(by_item):.1f}%) -- THIN, reported as such")
    print()

    # Per-model overall (marginal) wrongness rate, from ALL that model's
    # audited items (not just shared ones) -- a less circular "expected
    # under independence" baseline than using only the shared subsample.
    per_model_wrong = defaultdict(list)
    for item in key:
        w = scorer_wrong(item, answers)
        if w is not None:
            per_model_wrong[item["model"]].append(w)
    marginal_rate = {m: np.mean(v) for m, v in per_model_wrong.items() if len(v) > 0}

    # Build paired (model_i_wrong, model_j_wrong) observations from shared items.
    pairs = []
    for item_id, records in shared_items.items():
        wrongness = {}
        for r in records:
            w = scorer_wrong(r, answers)
            if w is not None:
                wrongness[r["model"]] = w
        models = list(wrongness.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                pairs.append({
                    "item_id": item_id, "model1": m1, "model2": m2,
                    "wrong1": wrongness[m1], "wrong2": wrongness[m2],
                    "expected1": marginal_rate.get(m1), "expected2": marginal_rate.get(m2),
                })

    print(f"Paired (model_i, model_j, same item) observations: {len(pairs)}")
    valid_pairs = [p for p in pairs if p["expected1"] is not None and p["expected2"] is not None]
    print(f"  valid (both models have a computable marginal rate): {len(valid_pairs)}")

    if len(valid_pairs) < 10:
        print()
        print("STOPPING: fewer than 10 valid paired observations. Any correlation")
        print("estimate here would be unreliable -- reporting this as an honest")
        print("null result on sample-size grounds, not as evidence of no correlation.")
        result = {"n_valid_pairs": len(valid_pairs), "conclusion": "insufficient_data"}
    else:
        w1 = np.array([1 if p["wrong1"] else 0 for p in valid_pairs])
        w2 = np.array([1 if p["wrong2"] else 0 for p in valid_pairs])
        exp1 = np.array([p["expected1"] for p in valid_pairs])
        exp2 = np.array([p["expected2"] for p in valid_pairs])

        observed_both_wrong = np.mean(w1 & w2)
        # expected under independence, using each pair's OWN two models'
        # marginal rates (not a single pooled rate -- more precise)
        expected_both_wrong_indep = np.mean(exp1 * exp2)

        print()
        print(f"Observed P(both models' scorer wrong | shared item): "
              f"{observed_both_wrong:.4f}")
        print(f"Expected under independence (product of each pair's own "
              f"marginals): {expected_both_wrong_indep:.4f}")
        ratio = observed_both_wrong / expected_both_wrong_indep if expected_both_wrong_indep > 0 else np.nan
        print(f"Ratio (observed / expected): {ratio:.3f}")

        # direct correlation coefficient on the paired binary indicators
        if len(set(w1)) > 1 and len(set(w2)) > 1:
            r, pval = pearsonr(w1, w2)
            print(f"Pearson correlation of (wrong1, wrong2) across shared items: "
                  f"r={r:.3f}, p={pval:.4f}")
        else:
            r, pval = None, None
            print("Pearson correlation not computable (no variance in one arm)")

        n_either_wrong = int(np.sum(w1 | w2))
        n_both_wrong = int(np.sum(w1 & w2))
        print(f"Raw event counts: {n_either_wrong}/{len(valid_pairs)} pairs have EITHER "
              f"model wrong; {n_both_wrong}/{len(valid_pairs)} have BOTH wrong.")

        print()
        print("=" * 70)
        if n_either_wrong < 5:
            verdict = "UNDERPOWERED"
            print(f"RESULT: UNDERPOWERED, not 'no correlation'. Only {n_either_wrong} of "
                  f"{len(valid_pairs)} paired observations have ANY wrongness event at "
                  f"all (consistent with E4's separate finding that beta is close to 0 "
                  f"almost everywhere on this roster). With this few events, correlation")
            print("cannot be estimated in either direction -- the ratio and correlation")
            print("coefficient above are not meaningful (0/0 or near-zero-variance).")
            print("This is a genuine current-audit-density limitation, not evidence for")
            print("or against T-C's cancellation mechanism. Do not report 'no")
            print("correlation found' -- report 'inconclusive, underpowered', and see")
            print("the cell-level complementary check below for a less direct but")
            print("higher-power test of the same underlying question.")
        elif ratio is not None and np.isfinite(ratio) and ratio > 1.2 and (pval is None or pval < 0.10):
            verdict = "POSITIVE_CORRELATION"
            print("RESULT: positive correlation detected -- shared item-difficulty")
            print("factor has empirical support. The naive independent-error bound")
            print("used in Phase 1/E-A is CONSERVATIVE for this reason: some of the")
            print("apparent differential width would cancel under a correlated model.")
        elif ratio is not None and np.isfinite(ratio) and ratio < 0.8:
            verdict = "NEGATIVE_CORRELATION"
            print("RESULT: NEGATIVE correlation -- scorer errors on the same item")
            print("across models are LESS likely to co-occur than independence would")
            print("predict. T-C's cancellation argument does NOT apply; if anything")
            print("this is evidence AGAINST conservatism from correlation.")
        else:
            verdict = "NO_CLEAR_CORRELATION"
            print("RESULT: no clear correlation (ratio near 1, or not statistically")
            print("distinguishable from independence at this sample size). T-C's")
            print("cancellation mechanism is NOT supported by this data -- the")
            print("independent-error model used in E-A is not shown to be")
            print("conservative on this basis. Report this plainly; do not claim")
            print("support for cancellation that the data does not show.")
        print("=" * 70)

        result = {
            "n_valid_pairs": len(valid_pairs),
            "n_either_wrong": n_either_wrong,
            "n_both_wrong": n_both_wrong,
            "observed_both_wrong": float(observed_both_wrong),
            "expected_both_wrong_independence": float(expected_both_wrong_indep),
            "ratio": float(ratio) if np.isfinite(ratio) else None,
            "pearson_r": float(r) if r is not None else None,
            "pearson_p": float(pval) if pval is not None else None,
            "verdict": verdict,
        }

    # ---- Complementary, higher-power test: cell-level (subject x quartile) ----
    # The exact-item-overlap test above is structurally limited to 46 items.
    # This test asks a related but weaker question with much more data (all
    # ~350 audited items, not just the 46 overlapping ones): does wrongness
    # rate vary significantly ACROSS (subject,quartile) cells, pooling all
    # models together? If cells differ significantly, that supports a shared,
    # non-model-specific difficulty channel existing at the cell level --
    # consistent with (but not proof of) the same mechanism T-C's item-level
    # test looked for directly. Explicitly weaker evidence, reported as such.
    print()
    print("-" * 70)
    print("COMPLEMENTARY TEST: does wrongness rate vary by (subject, quartile)")
    print("cell, pooling ALL models? (higher power, less direct than exact-item)")
    print("-" * 70)
    from scipy.stats import chi2_contingency

    cell_counts = defaultdict(lambda: [0, 0])  # cell -> [n_wrong, n_total]
    for item in key:
        w = scorer_wrong(item, answers)
        if w is None:
            continue
        cell = f"{item['subject']}|q{item['quartile']}"
        cell_counts[cell][1] += 1
        if w:
            cell_counts[cell][0] += 1

    cells = {c: v for c, v in cell_counts.items() if v[1] >= 3}  # need some n per cell
    print(f"Cells with >=3 scored items: {len(cells)} of {len(cell_counts)}")
    if len(cells) >= 4:
        table = np.array([[v[0], v[1] - v[0]] for v in cells.values()])
        # guard: chi2_contingency needs no all-zero rows/cols
        if table.sum() > 0 and (table.sum(axis=1) > 0).all():
            try:
                chi2, p_cell, dof, _ = chi2_contingency(table)
                print(f"Chi-square test across {len(cells)} cells: "
                      f"chi2={chi2:.2f}, dof={dof}, p={p_cell:.4f}")
                cell_verdict = ("CELLS_DIFFER" if p_cell < 0.05 else "CELLS_DO_NOT_DIFFER")
                if p_cell < 0.05:
                    print("RESULT: wrongness rate DOES vary significantly by cell --")
                    print("weak, indirect support for a shared difficulty channel at the")
                    print("subject/length level, consistent with (not proof of) T-C's")
                    print("proposed mechanism.")
                else:
                    print("RESULT: no significant cell-level variation detected at this")
                    print("sample size. No support found for a shared difficulty channel")
                    print("via this test either.")
            except ValueError as e:
                chi2, p_cell, cell_verdict = None, None, f"test_failed: {e}"
                print(f"Chi-square test could not run: {e}")
        else:
            chi2, p_cell, cell_verdict = None, None, "degenerate_table"
            print("Table degenerate (a cell has zero total or the whole table is empty)")
    else:
        chi2, p_cell, cell_verdict = None, None, "insufficient_cells"
        print("Fewer than 4 qualifying cells -- cannot run the test meaningfully.")

    cell_result = {
        "n_cells": len(cells),
        "chi2": float(chi2) if chi2 is not None else None,
        "p_value": float(p_cell) if p_cell is not None else None,
        "verdict": cell_verdict,
        "cell_wrongness_rates": {c: v[0] / v[1] for c, v in cells.items()},
    }

    with open("results/flipbudget/tc_correlated_error.json", "w") as f:
        json.dump({"item_level": result, "item_level_pairs": pairs,
                    "cell_level": cell_result}, f, indent=2)
    print("\nWritten to results/flipbudget/tc_correlated_error.json")
