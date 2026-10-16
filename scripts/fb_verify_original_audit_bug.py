"""Extends fb_verify_is_equiv_bug.py's check to the ORIGINAL 400-item
companion-project audit (mathhard_labelling_key.json / mathhard_l1.csv),
used by tc_correlated_error.py and ee_format_association.py
(TIER1_VERDICT.md's 1/41 and AUC=0.830 figures). Same bug (SIGALRM missing
on Windows, is_equiv always returns False, the ThreadPoolExecutor patch
does not fix it since the inner call still hits signal.signal internally)
-- checking whether it changed anything for THESE 400 items, not assumed
either way.

Run: python scripts/fb_verify_original_audit_bug.py
"""
import csv
import json
import sys
import types

sys.path.insert(0, "scripts")
if "datasets" not in sys.modules:
    _stub = types.ModuleType("datasets")
    _stub.Dataset = object
    sys.modules["datasets"] = _stub

from fb_verify_is_equiv_bug import matches_gold_fixed

import importlib.util
_spec = importlib.util.spec_from_file_location("leaderboard_math_utils", "vendor/leaderboard_math/utils.py")
mathutils_raw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mathutils_raw)


def is_indeterminate(answer):
    if answer is None or answer.strip() == "":
        return True
    return answer.strip().upper() in ("NONE", "CONTRADICTORY")


def matches_gold_broken(answer, gold):
    a = mathutils_raw.normalize_final_answer(answer)
    g = mathutils_raw.normalize_final_answer(gold)
    return a.strip() == g.strip() or mathutils_raw.is_equiv(a, g)  # is_equiv always False here


if __name__ == "__main__":
    print("=" * 70)
    print("Re-checking the ORIGINAL 400-item companion audit for the same bug")
    print("=" * 70)

    with open("results/analysis/mathhard_labelling_key.json", encoding="utf-8") as f:
        key_list = json.load(f)
    key = {item["uid"]: item for item in key_list}
    with open("results/analysis/mathhard_l1.csv", newline="", encoding="utf-8") as f:
        answers = {r["uid"]: r["your_answer"] for r in csv.DictReader(f)}

    n_usable, n_changed = 0, 0
    changed = []
    for uid, ans in answers.items():
        if is_indeterminate(ans) or uid not in key:
            continue
        n_usable += 1
        k = key[uid]
        gold = k["gold"]
        broken = matches_gold_broken(ans, gold)
        fixed = matches_gold_fixed(ans, gold)
        if broken != fixed:
            n_changed += 1
            scorer_correct = (k["stratum"] == "credited")
            changed.append({
                "uid": uid, "model": k.get("model"), "your_answer": ans, "gold": gold,
                "broken_correct": broken, "fixed_correct": fixed,
                "scorer_correct": scorer_correct,
                "scorer_wrong_broken": scorer_correct != broken,
                "scorer_wrong_fixed": scorer_correct != fixed,
            })

    print(f"Usable rows (400-item audit): {n_usable}")
    print(f"Rows where human_correct changes under the fixed comparator: {n_changed}")
    for r in changed[:30]:
        print(f"  uid={r['uid'][:12]} model={r['model']} answer={r['your_answer'][:35]!r} "
              f"gold={r['gold'][:35]!r} broken={r['broken_correct']} fixed={r['fixed_correct']}")
    if len(changed) > 30:
        print(f"  ... and {len(changed)-30} more")

    n_wrong_broken = sum(1 for uid, ans in answers.items() if not is_indeterminate(ans) and uid in key
                          and (key[uid]["stratum"] == "credited") != matches_gold_broken(ans, key[uid]["gold"]))
    n_wrong_fixed = sum(1 for uid, ans in answers.items() if not is_indeterminate(ans) and uid in key
                         and (key[uid]["stratum"] == "credited") != matches_gold_fixed(ans, key[uid]["gold"]))
    print(f"\nTotal scorer-wrong events, BROKEN comparator: {n_wrong_broken}")
    print(f"Total scorer-wrong events, FIXED comparator: {n_wrong_fixed}")
    print(f"(TIER1_VERDICT.md reported figure for the direct-overlap subtest: 1/41 paired events, "
          f"a different, paired-subset count -- {n_wrong_broken}/{n_usable} above is the FULL "
          f"400-item single-item wrongness count, not directly the same number, reported for context.)")

    out = {
        "n_usable_rows": n_usable, "n_rows_changed_by_fix": n_changed,
        "changed_rows": changed,
        "n_scorer_wrong_broken_comparator": n_wrong_broken,
        "n_scorer_wrong_fixed_comparator": n_wrong_fixed,
    }
    with open("results/flipbudget/original_audit_bug_verification.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWritten to results/flipbudget/original_audit_bug_verification.json")
