"""The three verifiers on hand-checked cases, including the Windows is_equiv bug case."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flipbudget import verifiers as v  # noqa: E402

BOXED_HALF = r"so the answer is \boxed{\frac{2}{4}}."
MINERVA_HALF = "Final Answer: The final answer is $\\frac{2}{4}$. I hope it is correct."


def test_sb_sympy_credits_equivalent_fraction():
    assert v.sb(BOXED_HALF, r"\frac{1}{2}")["verdict"] == 1


def test_sb_string_mode_reproduces_historical_windows_behaviour():
    assert v.sb(BOXED_HALF, r"\frac{1}{2}", mode="string")["verdict"] == 0
    assert v.sb(BOXED_HALF, r"\frac{2}{4}", mode="string")["verdict"] == 1


def test_sb_no_box_is_not_found():
    r = v.sb("the answer is 3", "3")
    assert r["verdict"] == 0 and r["found"] is False


def test_em_recorded_uses_minerva_phrase_not_box():
    assert v.em_recorded(MINERVA_HALF, r"\frac{1}{2}")["verdict"] == 1
    assert v.em_recorded(BOXED_HALF, r"\frac{1}{2}")["verdict"] == 0


def test_mv_equivalence_and_rejection():
    solution = r"Thus the value is $\boxed{\frac{1}{2}}$."
    assert v.mv(BOXED_HALF, solution)["verdict"] == 1
    assert v.mv(r"\boxed{3}", solution)["verdict"] == 0
