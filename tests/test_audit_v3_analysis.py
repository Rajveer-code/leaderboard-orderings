"""G3.5/G4.1: label aggregation and the preregistered falsification test, on synthetic labels."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fb_audit_v3_analysis import counts, errors, falsification  # noqa: E402

OP = json.load(open(ROOT / "results" / "flipbudget" / "g3_operating_point.json", encoding="utf-8"))
POOLED_EXISTING = {"000": [0, 223], "111": [142, 144]}  # existing audit, K=3 sympy (certify_results)


def labels(model, cell, n, n_correct, role="falsification"):
    return {f"{model}:{cell}:{i}": {"model": model, "cell": cell, "role": role, "in_overlap": False,
                                    "indeterminate": False, "file": "f.csv",
                                    "ystar": {"sympy": int(i < n_correct), "string": int(i < n_correct),
                                              "mv": int(i < n_correct)}}
            for i in range(n)}


def test_errors_direction():
    assert errors(x=90, n=100, cell="111") == 10   # all verifiers credit; 10 judged wrong
    assert errors(x=7, n=100, cell="000") == 7     # none credit; 7 judged right


def test_counts_pool_only_concordant_cells():
    L = labels("m", "111", 10, 8) | labels("m", "011", 6, 3, role="certification")
    pooled, per = counts(L, "sympy")
    assert pooled == {"111": [8, 10]} and per[("m", "011")] == [3, 6]


def test_counts_indeterminate_as_incorrect_vs_dropped():
    L = labels("m", "111", 4, 4)
    L["x"] = {"model": "m", "cell": "111", "role": "falsification", "in_overlap": False,
              "indeterminate": True, "file": "f.csv", "ystar": {r: None for r in ("sympy", "string", "mv")}}
    assert counts(L, "sympy")[0]["111"] == [4, 5]
    assert counts(L, "sympy", indet="drop")[0]["111"] == [4, 4]


def test_no_rejection_when_model_matches_the_pool():
    m = OP["falsification_models"][0]
    L = labels(m, "111", OP["n0_by_cell"]["111"], OP["n0_by_cell"]["111"] - 1)  # ~1% error, like the pool
    pooled, per = counts(L, "sympy")
    rejected, rows = falsification(per, pooled, POOLED_EXISTING, OP)
    assert not rejected and rows and all(r["margin"] <= OP["eps0"] for r in rows)


def test_rejection_when_one_model_is_far_worse():
    m = OP["falsification_models"][0]
    n = OP["n0_by_cell"]["111"]
    L = labels(m, "111", n, int(n * 0.6))  # 40% of this model's concordant-credit rows judged wrong
    pooled, per = counts(L, "sympy")
    rejected, rows = falsification(per, pooled, POOLED_EXISTING, OP)
    assert rejected and any(r["cell"] == "111" and r["reject"] for r in rows)


def test_rejection_threshold_is_no_worse_than_the_projection():
    """With the planned n0 and no other new labels, the leave-one-out pool equals the existing
    audit, so the error count that first rejects must match the operating point's projection."""
    m = OP["falsification_models"][0]
    for cell, n0 in OP["n0_by_cell"].items():
        first = next((e for e in range(n0 + 1)
                      if falsification(*counts(labels(m, cell, n0, n0 - e if cell == "111" else e), "sympy")[::-1],
                                       POOLED_EXISTING, OP)[0]), None)
        assert first == OP["reject_if_errors_at_least"][cell], (cell, first)


def test_a_bad_model_cannot_inflate_the_pool_it_is_tested_against():
    """Leave-one-out: the screened model's own errors must not raise its comparison bound."""
    m = OP["falsification_models"][0]
    n = OP["n0_by_cell"]["111"]
    _, rows = falsification(*counts(labels(m, "111", n, int(n * 0.6)), "sympy")[::-1], POOLED_EXISTING, OP)
    r = next(x for x in rows if x["cell"] == "111")
    assert r["pooled_n_excl_model"] == POOLED_EXISTING["111"][1] and r["reject"]


@pytest.mark.parametrize("ref", ["sympy", "string", "mv"])
def test_every_reference_routine_runs(ref):
    m = OP["falsification_models"][0]
    pooled, per = counts(labels(m, "000", 39, 0), ref)
    assert falsification(per, pooled, POOLED_EXISTING, OP)[1]
