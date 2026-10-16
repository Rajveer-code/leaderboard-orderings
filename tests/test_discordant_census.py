"""Amendment 6: the discordant census -- bounds, the D8 restoration rule, and the allocation.

Census bounds are not estimates, so they must (a) be exact when a domain is fully labelled,
(b) contain the true rate for every possible completion otherwise, and (c) add nothing to the
Sidak allocation.
"""
import gzip
import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget import certify as C  # noqa: E402
from fb_audit_v3_analysis import extraction_only  # noqa: E402


def test_census_bounds_exact_when_every_response_is_labelled():
    assert C.census_bounds(3, 10, 10) == (0.3, 0.3)
    assert C.census_bounds(0, 7, 7) == (0.0, 0.0)


@pytest.mark.parametrize("size", [1, 2, 5, 8])
def test_census_bounds_contain_every_completion(size):
    """Enumerate every truth for every labelled subset size: the bounds always contain it."""
    for truth in itertools.product((0, 1), repeat=size):
        for n in range(size + 1):
            x = sum(truth[:n])  # the first n responses are the labelled ones
            lo, hi = C.census_bounds(x, n, size)
            assert lo <= sum(truth) / size <= hi
            if n == size:
                assert lo == hi == sum(truth) / size


def test_census_bounds_width_is_the_unlabelled_share():
    lo, hi = C.census_bounds(2, 6, 10)
    assert hi - lo == pytest.approx(4 / 10)


@pytest.mark.parametrize("bad", [(-1, 3, 5), (4, 3, 5), (1, 6, 5), (0, 0, 0)])
def test_census_bounds_reject_impossible_counts(bad):
    with pytest.raises(ValueError):
        C.census_bounds(*bad)


def test_d8_restores_only_batches_whose_sole_failure_is_extraction():
    assert extraction_only(["extraction: 40/41 typed exactly as a regex extraction (p=1e-5)"])
    assert not extraction_only(["extraction: ...", "dwell: median 9.0 s < 15.0 s"])
    assert not extraction_only(["paste: 1 blocked paste/drop attempt(s)"])
    assert not extraction_only([])  # a passing batch is not "restored"


CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
AUDIT = ROOT / "results" / "flipbudget" / "audit_k3_map.json"


@pytest.mark.skipif(not (CENSUS.exists() and AUDIT.exists()), reason="census and audit map required")
def test_census_domains_add_no_sidak_parameter():
    import fb_certify as F
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    audit = json.load(open(AUDIT, encoding="utf-8"))
    key = {k["uid"]: k for k in json.load(open(ROOT / "results/analysis/mathhard_labelling_key.json",
                                                encoding="utf-8"))}
    for it in audit["items"]:
        it["subject"], it["quartile"] = key[it["uid"]]["subject"], key[it["uid"]]["quartile"]
    models = census["models"]
    base = F.Inputs(census, audit, 3, "sympy", "sympy", "as_incorrect", models, "kl")
    m, c = models[0], next(iter(base.disc))
    sampled = {"disc": {(m, c): [2, 5]}}
    with_sample = F.Inputs(census, audit, 3, "sympy", "sympy", "as_incorrect", models, "kl", new_labels=sampled)
    with_census = F.Inputs(census, audit, 3, "sympy", "sympy", "as_incorrect", models, "kl", new_labels=sampled,
                           disc_fixed={(m, c): (0.4, 0.4)})
    assert with_sample.n_params() == base.n_params() + 1   # an estimated rate is one more parameter
    assert with_census.n_params() == base.n_params()       # a census bound is not a parameter
    assert with_census.disc_bounds(m, 0.99)[c] == (0.4, 0.4)
