import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fb_labeller_agreement import cohen_kappa  # noqa: E402


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == 0.5  # po .75, pe .5
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) + 1.0) < 1e-12


def test_baseline_matches_the_main_certification():
    """The labeller script must start from exactly the counts the paper reports."""
    la = json.load(open(ROOT / "results" / "flipbudget" / "labeller_agreement.json", encoding="utf-8"))
    cert = json.load(open(ROOT / "results" / "flipbudget" / "certify_results.json", encoding="utf-8"))["K3"]["fixed_board"]
    assert la["certified"]["baseline"]["eps0"] == cert["certified_eps0"]
    assert la["certified"]["baseline"]["prereg"] == cert["certified_eps_preregistered"]


def test_overlap_sizes_are_the_preregistered_ones():
    la = json.load(open(ROOT / "results" / "flipbudget" / "labeller_agreement.json", encoding="utf-8"))
    assert (la["y_star"]["two"]["n"], la["y_star"]["three"]["n"]) == (151, 75)
    assert la["string_level"]["l1_vs_l2"]["n_shared"] == 151 and la["string_level"]["l1_vs_l3"]["n_shared"] == 75
