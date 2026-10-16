"""The working MATH-Hard comparator. `is_equiv`/`matches_gold` elsewhere in
this project (fb_tc_real_analysis.py, tc_correlated_error.py,
ee_format_association.py) rely on the vendor's own `timeout()` context
manager, which calls `signal.signal(signal.SIGALRM, ...)` -- SIGALRM does
not exist on Windows (confirmed: hasattr(signal,'SIGALRM')==False on this
machine), so is_equiv's own broad `except Exception: return False` silently
swallows the resulting AttributeError and returns False for EVERY call,
including genuinely equivalent expressions. The existing
`ThreadPoolExecutor`-based "fix" in the three scripts above does not
actually fix this: it wraps the same broken call, whose OWN internal
exception handler fires before the wrapper's timeout logic is ever reached.

Root-caused and verified in `fb_verify_is_equiv_bug.py` /
`fb_verify_original_audit_bug.py` before writing this: T-C's own 69-row
result is unaffected (0/0 rows change); the original 400-item companion
audit had exactly 1 row affected (5->6 true scorer-wrong events).

This module provides the actual fix: the same parse/subtract/simplify core
logic, timed out via ThreadPoolExecutor alone, with NO signal.alarm
dependency anywhere in the call path.

Import `matches_gold_fixed`/`is_equiv_fixed` for any NEW analysis. Existing
committed results are not retroactively rewritten by this file -- see
MATH_COMPARATOR_BUG.md for what was and was not affected.
"""
import importlib.util
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import sympy
from sympy.parsing.latex import parse_latex

if "datasets" not in sys.modules:
    _stub = types.ModuleType("datasets")
    _stub.Dataset = object
    sys.modules["datasets"] = _stub

_spec = importlib.util.spec_from_file_location("leaderboard_math_utils", "vendor/leaderboard_math/utils.py")
mathutils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mathutils)

_POOL = ThreadPoolExecutor(max_workers=4)


def _is_equiv_core(x1, x2):
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


def is_indeterminate(answer):
    if answer is None or answer.strip() == "":
        return True
    return answer.strip().upper() in ("NONE", "CONTRADICTORY")


def matches_gold_fixed(answer, gold):
    a = mathutils.normalize_final_answer(answer)
    g = mathutils.normalize_final_answer(gold)
    return a.strip() == g.strip() or is_equiv_fixed(a, g)


if __name__ == "__main__":
    assert hasattr(sys.modules[__name__], "matches_gold_fixed")
    cases = [
        (r"\frac{2}{4}", r"\frac{1}{2}", True),
        ("x+3", "3+x", True),
        (r"-\frac{1}{2}", r"\frac{-1}{2}", True),
        (r"\frac{1}{3}", r"\frac{1}{2}", False),
    ]
    ok = all(matches_gold_fixed(a, g) == expected for a, g, expected in cases)
    print(f"[{'OK' if ok else 'FAIL'}] self-check: {ok}")
    if not ok:
        raise SystemExit(1)
