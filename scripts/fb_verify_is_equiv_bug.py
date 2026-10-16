"""URGENT verification, found via the adversarial suite: is_equiv() appears to
ALWAYS return False on this machine, for every input, including genuinely
equivalent expressions. Root cause: is_equiv's own timeout wrapper uses
signal.SIGALRM, which does not exist on Windows (confirmed:
hasattr(signal,'SIGALRM')==False here); is_equiv's own broad
`except Exception: return False` silently swallows the resulting
AttributeError. fb_tc_real_analysis.py's ThreadPoolExecutor patch does NOT
fix this -- it wraps the same broken call, whose own internal exception
handler still fires before the wrapper's timeout can matter.

This script: (1) proves the bug directly, (2) builds a working replacement
that does not depend on signal.alarm, (3) re-checks all 69 real T-C rows to
see whether human_correct/scorer_wrong actually changes for any of them.

Run: python scripts/fb_verify_is_equiv_bug.py
"""
import json
import signal
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

sys.path.insert(0, "scripts")
if "datasets" not in sys.modules:
    _stub = types.ModuleType("datasets")
    _stub.Dataset = object
    sys.modules["datasets"] = _stub

import importlib.util
_spec = importlib.util.spec_from_file_location("leaderboard_math_utils", "vendor/leaderboard_math/utils.py")
mathutils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mathutils)

import sympy
from sympy.parsing.latex import parse_latex

_POOL = ThreadPoolExecutor(max_workers=4)


def _is_equiv_core(x1, x2):
    """The SAME core logic as the vendor's is_equiv (parse, subtract,
    simplify), with NO signal.alarm dependency -- the timeout is provided
    entirely by the ThreadPoolExecutor wrapper below."""
    try:
        p1, p2 = parse_latex(x1), parse_latex(x2)
    except Exception:
        return False
    try:
        diff = p1 - p2
    except Exception:
        return False
    try:
        return bool(sympy.simplify(diff) == 0)
    except Exception:
        return False


def is_equiv_fixed(x1, x2, seconds=2.0):
    fut = _POOL.submit(_is_equiv_core, x1, x2)
    try:
        return fut.result(timeout=seconds)
    except FutureTimeout:
        return False


def matches_gold_fixed(answer, gold):
    a = mathutils.normalize_final_answer(answer)
    g = mathutils.normalize_final_answer(gold)
    return a.strip() == g.strip() or is_equiv_fixed(a, g)


if __name__ == "__main__":
    print("=" * 70)
    print("Verifying and fixing the is_equiv Windows SIGALRM bug")
    print("=" * 70)

    print(f"\nhasattr(signal, 'SIGALRM'): {hasattr(signal, 'SIGALRM')}")

    print("\n--- Confirming the bug in the ORIGINAL (broken-patch) is_equiv ---")
    from fb_tc_real_analysis import matches_gold as matches_gold_broken

    broken_cases = [
        (r"\frac{2}{4}", r"\frac{1}{2}", True, "unsimplified fraction"),
        ("x+3", "3+x", True, "commutative reorder"),
        (r"-\frac{1}{2}", r"\frac{-1}{2}", True, "sign placement"),
    ]
    n_bug_confirmed = 0
    for a, g, expected, tag in broken_cases:
        broken_result = matches_gold_broken(a, g)
        fixed_result = matches_gold_fixed(a, g)
        bug_present = (broken_result != expected) and (fixed_result == expected)
        n_bug_confirmed += bug_present
        print(f"  {tag}: broken={broken_result} fixed={fixed_result} expected={expected} "
              f"[{'BUG CONFIRMED' if bug_present else 'no bug here'}]")

    print(f"\nBug confirmed on {n_bug_confirmed}/{len(broken_cases)} known-equivalent constructed cases.")

    print("\n--- Re-checking all 69 real T-C rows with the FIXED comparator ---")
    from fb_tc_real_analysis import load_l1, load_key, is_indeterminate

    key = load_key()
    l1_answers = load_l1()

    n_changed = 0
    changed_rows = []
    n_usable = 0
    for uid, ans in l1_answers.items():
        if is_indeterminate(ans):
            continue
        n_usable += 1
        k = key[uid]
        human_correct_broken = matches_gold_broken(ans, k["gold"])
        human_correct_fixed = matches_gold_fixed(ans, k["gold"])
        if human_correct_broken != human_correct_fixed:
            n_changed += 1
            scorer_correct = (k["stratum"] == "credited")
            changed_rows.append({
                "uid": uid, "model": k["model"], "your_answer": ans, "gold": k["gold"],
                "human_correct_broken": human_correct_broken, "human_correct_fixed": human_correct_fixed,
                "scorer_correct": scorer_correct,
                "scorer_wrong_broken": scorer_correct != human_correct_broken,
                "scorer_wrong_fixed": scorer_correct != human_correct_fixed,
            })

    print(f"Usable rows checked: {n_usable}")
    print(f"Rows where human_correct changes under the fix: {n_changed}")
    for r in changed_rows:
        print(f"  uid={r['uid'][:12]} answer={r['your_answer'][:40]!r} gold={r['gold'][:40]!r} "
              f"broken_correct={r['human_correct_broken']} fixed_correct={r['human_correct_fixed']} "
              f"scorer_wrong: broken={r['scorer_wrong_broken']} fixed={r['scorer_wrong_fixed']}")

    n_scorer_wrong_fixed = sum(1 for r in changed_rows if r["scorer_wrong_fixed"]) + \
        sum(1 for uid, ans in l1_answers.items() if not is_indeterminate(ans)
            and uid not in {c["uid"] for c in changed_rows}
            and (key[uid]["stratum"] == "credited") != matches_gold_fixed(ans, key[uid]["gold"]))

    print(f"\nTotal scorer-wrong events under the FIXED comparator (all 69 rows): {n_scorer_wrong_fixed}")
    print(f"(Original reported figure, broken comparator: 0/69)")

    out = {
        "sigalrm_present": hasattr(signal, "SIGALRM"),
        "bug_confirmed_on_constructed_cases": n_bug_confirmed,
        "n_constructed_cases": len(broken_cases),
        "n_real_rows_checked": n_usable,
        "n_real_rows_changed": n_changed,
        "changed_rows": changed_rows,
        "n_scorer_wrong_under_fixed_comparator": n_scorer_wrong_fixed,
        "n_scorer_wrong_under_broken_comparator_original": 0,
    }
    with open("results/flipbudget/is_equiv_bug_verification.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWritten to results/flipbudget/is_equiv_bug_verification.json")
