"""E4 -- per-model differential misclassification on MATH-Hard, real data,
with partial pooling per PREREGISTRATION_FLIPBUDGET.md section 2.

Joins mathhard_labelling_key.json (400 items, per-item model + stratum, where
stratum encodes the SCORER's verdict: credited=scorer said correct,
wrong_parsed/unparsed=scorer said incorrect/unparseable) against
mathhard_l1.csv (the human labeller's own answer per item, "your_answer") to
recover, per model:
  alpha_j = P(scorer credits | human says wrong)  -- estimated within the
            credited stratum, as (items where your_answer != gold) / n_credited
  beta_j  = P(scorer misses | human says right)   -- estimated within the
            wrong_parsed+unparsed strata, as (your_answer == gold) / n_wrong

VERIFICATION STRATEGY: this project's own comparator ("harness
normalize_final_answer + is_equiv", per mathhard_human_margins.json) is not
directly imported here (not cleanly separable from scripts/19's network/CLI
code without more exploration than the remaining budget allows). Instead: a
standard MATH-benchmark equivalence check (Hendrycks et al. normalization,
the same normalization used broadly across lm-evaluation-harness's MATH
task) is implemented below, and the FIRST thing this script does is
reproduce the already-published POOLED result from mathhard_human_margins.json
(credited-stratum false-credit rate 0.0314, n_used=162 of 175) as a
correctness check on the comparator before trusting any per-model breakdown.
If that check fails, the script stops -- it does not proceed to report
per-model numbers on an unverified comparator.

Run: python scripts/fb_e4_mathhard.py
"""
import csv
import json
import importlib.util
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

# Use the ACTUAL comparator this project's MATH-Hard scoring is built on --
# lm_eval/tasks/leaderboard/math/utils.py (sympy + math-verify LaTeX parsing),
# loaded the exact way scripts/19::load_math_utils does it, not a
# reimplementation. Two earlier attempts got this wrong: a hand-rolled string
# normalizer (0.171 vs published 0.031, 5.5x off), then the WRONG vendored
# file (hendrycks_math/utils.py, an older/simpler comparator -- also off).
# Reimplementing MATH-answer equivalence from scratch is exactly the kind of
# thing that produces a subtly wrong, silently plausible number; importing
# the real, already-tuned function this project's own numbers are built on
# is the correct fix, not a shortcut.
if "datasets" not in sys.modules:
    _stub = types.ModuleType("datasets")
    _stub.Dataset = object
    sys.modules["datasets"] = _stub

_here = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "leaderboard_math_utils", _here / "vendor" / "leaderboard_math" / "utils.py")
_mathutils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mathutils)

# is_equiv's sympy LaTeX parse guards itself with SIGALRM, which does not
# exist on Windows -- restore the same ThreadPoolExecutor-based timeout
# wrapper scripts/19::load_math_utils uses, faithfully, not stubbed to a
# no-op (an earlier run of THAT script hung indefinitely when someone tried
# that shortcut -- see its own comment).
if not hasattr(__import__("signal"), "SIGALRM"):
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout
    _original_is_equiv = _mathutils.is_equiv
    _cache: dict = {}
    _pool = ThreadPoolExecutor(max_workers=4)

    def _is_equiv_with_deadline(x1: str, x2: str, seconds: float = 1.0) -> bool:
        key = (x1, x2)
        if key in _cache:
            return _cache[key]
        fut = _pool.submit(_original_is_equiv, x1, x2)
        try:
            result = fut.result(timeout=seconds)
        except FutureTimeout:
            result = False
        _cache[key] = result
        return result

    _mathutils.is_equiv = _is_equiv_with_deadline


def is_indeterminate(answer: str) -> bool:
    """Matches scripts/64_score_mathhard_human.py exactly: None/empty or the
    literal sentinels NONE/CONTRADICTORY (case-insensitive) mean the labeller
    could not or would not commit the response to an answer."""
    if answer is None or answer.strip() == "":
        return True
    return answer.strip().upper() in ("NONE", "CONTRADICTORY")


def matches_gold(answer: str, gold: str) -> bool:
    """Exact mirror of scripts/64_score_mathhard_human.py's matches_gold: both
    sides pre-normalized via normalize_final_answer, then exact match OR
    is_equiv -- not a re-derivation, a direct copy of the already-correct
    logic, after an earlier version of this script got this wrong (called
    is_equiv on raw un-normalized strings and did not exclude NONE/
    CONTRADICTORY as indeterminate, inflating the false-credit rate 5x)."""
    a = _mathutils.normalize_final_answer(answer)
    g = _mathutils.normalize_final_answer(gold)
    return a.strip() == g.strip() or _mathutils.is_equiv(a, g)


def load_data():
    with open("results/analysis/mathhard_labelling_key.json") as f:
        key = json.load(f)
    with open("results/analysis/mathhard_l1.csv", newline="", encoding="utf-8") as f:
        answers = {row["uid"]: row["your_answer"] for row in csv.DictReader(f)}
    return key, answers


def pooled_verification_check(key, answers):
    """Reproduce mathhard_human_margins.json's published credited-stratum
    false-credit rate (0.0314, n_used=162 of 175) before trusting anything
    else in this script."""
    credited = [item for item in key if item["stratum"] == "credited"]
    n_total = len(credited)
    n_used, n_false_credit = 0, 0
    for item in credited:
        ans = answers.get(item["uid"])
        if is_indeterminate(ans):
            continue
        n_used += 1
        if not matches_gold(ans, item["gold"]):
            n_false_credit += 1  # scorer credited it, human says the answer doesn't match gold
    rate = n_false_credit / n_used if n_used else float("nan")
    print(f"Pooled verification check (credited stratum):")
    print(f"  n_total_in_stratum={n_total} (published: 175)")
    print(f"  n_used={n_used} (published: 162)")
    print(f"  false-credit rate={rate:.4f} (published: 0.0314)")
    close = abs(rate - 0.03137299381840867) < 0.01 and abs(n_used - 162) <= 5
    print(f"  [{'OK' if close else 'FAIL'}] comparator reproduces the published pooled rate "
          f"within tolerance: {close}")
    return close


def per_model_rates(key, answers):
    """Returns {model: {alpha_raw, beta_raw, n_credited, n_wrong, n_used_credited,
    n_used_wrong}}"""
    by_model = defaultdict(lambda: {"credited": [], "wrong": []})
    for item in key:
        m = item["model"]
        if item["stratum"] == "credited":
            by_model[m]["credited"].append(item)
        else:  # wrong_parsed or unparsed
            by_model[m]["wrong"].append(item)

    out = {}
    for model, groups in by_model.items():
        cred = groups["credited"]
        n_cred = len(cred)
        n_used_cred = 0
        n_false_credit = 0
        for item in cred:
            ans = answers.get(item["uid"])
            if is_indeterminate(ans):
                continue
            n_used_cred += 1
            if not matches_gold(ans, item["gold"]):
                n_false_credit += 1
        alpha_raw = n_false_credit / n_used_cred if n_used_cred else np.nan

        wrong = groups["wrong"]
        n_wrong = len(wrong)
        n_used_wrong = 0
        n_false_miss = 0
        for item in wrong:
            ans = answers.get(item["uid"])
            if is_indeterminate(ans):
                continue
            n_used_wrong += 1
            if matches_gold(ans, item["gold"]):
                n_false_miss += 1
        beta_raw = n_false_miss / n_used_wrong if n_used_wrong else np.nan

        out[model] = {
            "alpha_raw": alpha_raw, "n_credited": n_cred, "n_used_credited": n_used_cred,
            "beta_raw": beta_raw, "n_wrong": n_wrong, "n_used_wrong": n_used_wrong,
            "n_total_audited": n_cred + n_wrong,
        }
    return out


def partial_pool(per_model, pooled_alpha, pooled_beta):
    """Empirical-Bayes shrinkage: each model's raw rate shrinks toward the
    pooled rate, weighted by relative precision. Simple, standard James-Stein-
    style form: weight = n_model / (n_model + k), k chosen as the pooled
    stratum's own effective n (i.e. a model needs roughly that many audited
    items before its own data outweighs the pooled prior)."""
    k_alpha = 162  # published pooled n_used for the credited stratum
    k_beta = 167  # published pooled n_used for the wrong_parsed stratum (from
    # mathhard_human_margins.json's earlier printout)
    pooled = {}
    for model, d in per_model.items():
        n_a = d["n_used_credited"]
        w_a = n_a / (n_a + k_alpha) if n_a > 0 else 0.0
        alpha_raw = d["alpha_raw"] if not np.isnan(d["alpha_raw"]) else pooled_alpha
        alpha_pooled = w_a * alpha_raw + (1 - w_a) * pooled_alpha

        n_b = d["n_used_wrong"]
        w_b = n_b / (n_b + k_beta) if n_b > 0 else 0.0
        beta_raw = d["beta_raw"] if not np.isnan(d["beta_raw"]) else pooled_beta
        beta_pooled = w_b * beta_raw + (1 - w_b) * pooled_beta

        pooled[model] = {
            **d,
            "alpha_pooled": alpha_pooled, "shrinkage_weight_alpha": w_a,
            "beta_pooled": beta_pooled, "shrinkage_weight_beta": w_b,
        }
    return pooled


if __name__ == "__main__":
    print("=" * 70)
    print("E4 -- MATH-Hard per-model differential misclassification")
    print("=" * 70)
    key, answers = load_data()
    print(f"Loaded {len(key)} labelled items, {len(answers)} human answers")
    print()

    ok = pooled_verification_check(key, answers)
    print()
    if not ok:
        print("STOPPING: comparator does not reproduce the published pooled rate")
        print("within tolerance. Do not trust per-model numbers below without")
        print("fixing the comparator first.")
        raise SystemExit(1)

    print("=" * 70)
    print("Per-model raw rates (real data, no pooling)")
    print("=" * 70)
    per_model = per_model_rates(key, answers)
    with open("results/analysis/mathhard_human_margins.json") as f:
        mm = json.load(f)
    pooled_alpha = mm["strata"]["credited"]["primary"]["rate_within_stratum"]
    pooled_beta = mm["strata"]["wrong_parsed"]["primary"]["rate_within_stratum"]
    print(f"(pooled anchor: alpha={pooled_alpha:.4f}, beta={pooled_beta:.4f})")
    print()

    pooled_result = partial_pool(per_model, pooled_alpha, pooled_beta)

    rows = sorted(pooled_result.items(), key=lambda kv: -kv[1]["n_total_audited"])
    print(f"{'model':<45} {'n_aud':>6} {'a_raw':>7} {'a_pool':>7} {'b_raw':>7} {'b_pool':>7} {'w_a':>5}")
    for model, d in rows:
        a_raw = d["alpha_raw"] if not np.isnan(d["alpha_raw"]) else float("nan")
        b_raw = d["beta_raw"] if not np.isnan(d["beta_raw"]) else float("nan")
        print(f"{model[:44]:<45} {d['n_total_audited']:>6} {a_raw:>7.3f} "
              f"{d['alpha_pooled']:>7.3f} {b_raw:>7.3f} {d['beta_pooled']:>7.3f} "
              f"{d['shrinkage_weight_alpha']:>5.2f}")

    # Kill Criterion 6 check: does pooling actually separate models, or does
    # everything collapse to the pooled anchor?
    alphas_pooled = np.array([d["alpha_pooled"] for _, d in rows])
    betas_pooled = np.array([d["beta_pooled"] for _, d in rows])
    print()
    print(f"Spread after pooling: alpha std={alphas_pooled.std():.4f} "
          f"(range {alphas_pooled.min():.3f}-{alphas_pooled.max():.3f}), "
          f"beta std={betas_pooled.std():.4f} "
          f"(range {betas_pooled.min():.3f}-{betas_pooled.max():.3f})")
    separated = alphas_pooled.std() > 0.005 or betas_pooled.std() > 0.01
    print(f"[{'OK -- models separate after pooling' if separated else 'KILL CRITERION 6 -- pooling collapses to the anchor'}]")

    out = {
        "verification_check_passed": ok,
        "pooled_anchor": {"alpha": pooled_alpha, "beta": pooled_beta},
        "shrinkage_k": {"alpha": 162, "beta": 167},
        "per_model": {m: {k2: (None if isinstance(v2, float) and np.isnan(v2) else v2)
                           for k2, v2 in d.items()} for m, d in pooled_result.items()},
        "kill_criterion_6_triggered": bool(not separated),
    }
    import os
    os.makedirs("results/flipbudget", exist_ok=True)
    with open("results/flipbudget/e4_mathhard_per_model.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Written to results/flipbudget/e4_mathhard_per_model.json")
