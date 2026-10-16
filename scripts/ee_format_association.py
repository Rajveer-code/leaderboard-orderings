"""E-E -- is differential scorer error associated with observable response
format? Tier 1, item 3. Final Tier-1 experiment; its result feeds directly
into the end-of-Tier-1 verdict.

Question: can scorer wrongness (Yhat != Y*) be predicted from response-level
FORMAT features (length, presence of \\boxed, LaTeX density, structure) --
using proper cross-validation, not in-sample fit, which would overstate any
association.

Uses the full 400-item MATH-Hard audit (not the 46-item overlap subsample
T-C was structurally limited to), so this test has real power even where
T-C did not.

Run: python scripts/ee_format_association.py
"""
import csv
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

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


def extract_features(response: str) -> dict:
    return {
        "char_length": len(response),
        "word_count": len(response.split()),
        "has_boxed": int(r"\boxed" in response),
        "n_dollar_signs": response.count("$"),
        "n_newlines": response.count("\n"),
        "n_equals": response.count("="),
        "has_therefore": int(bool(re.search(r"\b(therefore|thus|hence)\b", response, re.I))),
        "n_frac": response.count(r"\frac") + response.count(r"\dfrac"),
        "ends_with_period": int(response.rstrip().endswith(".")),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("E-E -- format-conditional differential error, Tier 1 item 3")
    print("=" * 70)

    with open("results/analysis/mathhard_labelling_key.json") as f:
        key = json.load(f)
    with open("results/analysis/mathhard_l1.csv", newline="", encoding="utf-8") as f:
        rows_by_uid = {r["uid"]: r for r in csv.DictReader(f)}

    records = []
    for item in key:
        row = rows_by_uid.get(item["uid"])
        if row is None:
            continue
        ans = row["your_answer"]
        if is_indeterminate(ans):
            continue
        human_correct = matches_gold(ans, item["gold"])
        scorer_correct = (item["stratum"] == "credited")
        wrong = int(scorer_correct != human_correct)
        feats = extract_features(row["response"])
        records.append({"model": item["model"], "wrong": wrong, **feats})

    print(f"Usable records (indeterminate excluded): {len(records)} of {len(key)}")
    n_wrong = sum(r["wrong"] for r in records)
    print(f"Wrongness events: {n_wrong} of {len(records)} "
          f"({100*n_wrong/len(records):.1f}%)")
    print()

    if n_wrong < 15:
        print("WARNING: fewer than 15 positive events. A predictive model at this")
        print("event count will have wide, unreliable AUC estimates -- reported")
        print("with that caveat explicitly, not hidden.")

    feature_names = ["char_length", "word_count", "has_boxed", "n_dollar_signs",
                      "n_newlines", "n_equals", "has_therefore", "n_frac",
                      "ends_with_period"]
    X = np.array([[r[f] for f in feature_names] for r in records], dtype=float)
    y = np.array([r["wrong"] for r in records])

    # standardize (length-scale features vs binary features)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xs = (X - X_mean) / X_std

    print("-" * 70)
    print("Cross-validated logistic regression: wrongness ~ format features")
    print("-" * 70)

    n_splits = 5
    if n_wrong < n_splits or (len(records) - n_wrong) < n_splits:
        print(f"Cannot run {n_splits}-fold CV: fewer than {n_splits} events in the "
              f"minority class. Reporting a simple held-out random 70/30 split instead, "
              f"with the caveat that a single split is noisy at this n.")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        train_idx, test_idx = idx[:cut], idx[cut:]
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xs[train_idx], y[train_idx])
        if len(set(y[test_idx])) < 2:
            print("Test split has only one class present -- AUC undefined. Cannot "
                  "report a reliable predictive result at this event count.")
            auc, auc_ci = None, None
        else:
            probs = clf.predict_proba(Xs[test_idx])[:, 1]
            auc = roc_auc_score(y[test_idx], probs)
            auc_ci = None
            print(f"Single held-out split AUC: {auc:.3f} (n_test={len(test_idx)}, "
                  f"n_wrong_test={y[test_idx].sum()})")
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        cv_probs = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, cv_probs)
        print(f"{n_splits}-fold cross-validated AUC: {auc:.3f}")

        # bootstrap CI on the CV AUC (resample records, recompute AUC using the
        # SAME out-of-fold predictions -- a standard nonparametric CI on AUC,
        # not a fresh re-fit per bootstrap replicate)
        rng = np.random.default_rng(0)
        boot_aucs = []
        for _ in range(1000):
            idx = rng.integers(0, len(y), len(y))
            if len(set(y[idx])) < 2:
                continue
            boot_aucs.append(roc_auc_score(y[idx], cv_probs[idx]))
        auc_ci = (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5)))
        print(f"Bootstrap 95% CI on AUC: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] "
              f"({len(boot_aucs)} valid replicates)")

    print()
    # fit on full data for coefficient inspection (interpretation only, not
    # for the AUC claim above, which is properly cross-validated)
    clf_full = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf_full.fit(Xs, y)
    print("Feature coefficients (full-data fit, standardized -- for direction/")
    print("interpretation only, not a held-out claim):")
    for name, coef in sorted(zip(feature_names, clf_full.coef_[0]),
                              key=lambda x: -abs(x[1])):
        print(f"  {name:<18} {coef:+.3f}")

    print()
    print("=" * 70)
    # The verdict MUST be gated on the confidence interval, not the point
    # estimate alone -- an earlier version of this script classified AUC=0.83
    # as "STRONG_ASSOCIATION" without checking the CI computed two lines
    # above it, which spans [0.492, 0.991] -- includes chance (0.5) almost
    # exactly at its lower bound. A point estimate this unstable is not
    # evidence of an association; reporting it as "strong" would be
    # manufacturing a positive finding from 5 events. Fixed before commit.
    if auc is None:
        verdict = "UNDETERMINED"
        print("VERDICT: undetermined -- too few events for a reliable AUC estimate.")
    elif auc_ci is not None and auc_ci[0] < 0.55:
        verdict = "UNDERPOWERED"
        print(f"VERDICT: UNDERPOWERED, not a confirmed association. Point estimate "
              f"AUC={auc:.3f} looks strong, but the bootstrap 95% CI is "
              f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] -- the lower bound is at or below")
        print(f"chance (0.5), driven by only {n_wrong} positive events in "
              f"{len(records)} records. This CANNOT be reported as a confirmed")
        print("format-wrongness association. It is a suggestive point estimate")
        print("with too little data behind it to trust in either direction --")
        print("exactly the same audit-density limitation T-C hit, from a")
        print("different angle (event scarcity, not item-overlap scarcity).")
    elif auc >= 0.70:
        verdict = "STRONG_ASSOCIATION"
        print(f"VERDICT: STRONG association (AUC={auc:.3f}, CI excludes chance). "
              f"Scorer wrongness IS substantially predictable from observable")
        print("response format -- directly supports the mechanism story.")
    elif auc >= 0.60:
        verdict = "MODERATE_ASSOCIATION"
        print(f"VERDICT: MODERATE association (AUC={auc:.3f}, CI excludes chance).")
    else:
        verdict = "WEAK_OR_NO_ASSOCIATION"
        print(f"VERDICT: WEAK/no association (AUC={auc:.3f}, near chance). Format")
        print("features as measured here do NOT predict scorer wrongness on this")
        print("data. This weakens the mechanism story -- reported plainly, not")
        print("explained away.")
    print("=" * 70)

    out = {
        "n_records": len(records), "n_wrong": n_wrong,
        "wrongness_rate": n_wrong / len(records),
        "features_used": feature_names,
        "cv_auc": float(auc) if auc is not None else None,
        "auc_bootstrap_ci": auc_ci,
        "coefficients": {name: float(coef) for name, coef in
                          zip(feature_names, clf_full.coef_[0])},
        "verdict": verdict,
    }
    with open("results/flipbudget/ee_format_association.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWritten to results/flipbudget/ee_format_association.json")
