"""T21-T23: the template predicate, the taxonomy classifier, and the meaning-preserving rewrites."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget import verifiers as V  # noqa: E402
from fb_extraction_taxonomy import classify  # noqa: E402
from fb_perturbation import rewrites, strip_boxes  # noqa: E402

OPENING = r"Kathy has $\boxed{22}$ more euros than pounds. Final Answer: The final answer is 22. Great job!"
TEMPLATE = r"We get $\boxed{7}$. Final Answer: The final answer is 7. I hope it is correct."


def test_template_predicate_matches_the_papers_opening_example():
    assert not V.em_template_found(OPENING)          # "Great job!" breaks the template
    assert V.em_template_found(TEMPLATE)
    assert not V.em_template_found(r"So the answer is $\boxed{22}$.")


def test_classifier_is_exhaustive_and_names_the_failed_extractor():
    base = {"em": 0, "sb_sympy": 1, "mv": 1, "sb_found": True, "mv_pred_parsed": True}
    assert classify(base, template_found=False) == "extraction:template"
    assert classify(base, template_found=True) == "judgement"
    box = {"em": 1, "sb_sympy": 0, "mv": 1, "sb_found": False, "mv_pred_parsed": True}
    assert classify(box, template_found=True) == "extraction:box"
    assert classify(base | {"mv_timeout": True}, template_found=True) == "runtime"


def test_rewrites_keep_the_answer_and_change_only_surface():
    assert strip_boxes(r"a \boxed{\frac{1}{2}} b \boxed{3}") == r"a \frac{1}{2} b 3"
    r = rewrites(TEMPLATE)
    assert r["signoff"].endswith("is 7. Great job!")
    assert r["no_template"].strip() == r"We get $\boxed{7}$."
    assert "7" in r["no_box"] and r"\boxed" not in r["no_box"]
    assert r["control"].rstrip() == TEMPLATE
    assert "no_box" not in rewrites("Final Answer: The final answer is 7. I hope it is correct.")
