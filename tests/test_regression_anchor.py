"""G1.6 regression anchor.

On the old data (17 models, K=2 cells in the historical string semantics, the existing
400-item audit) the certification code must reproduce 84 combined / 101 audit-only /
118 sampling-ceiling orderings. The inputs are rebuilt from the NEW census and audit map,
so this also checks that the new data path reproduces the old one exactly.
"""
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
AUDIT_MAP = ROOT / "results" / "flipbudget" / "audit_k3_map.json"

pytestmark = pytest.mark.skipif(not (CENSUS.exists() and AUDIT_MAP.exists()),
                                reason="run scripts/fb_census_k3.py and scripts/fb_audit_k3_map.py first")


@pytest.fixture(scope="module")
def anchor_inputs():
    import fb_joint_verifier as jv

    audit_map = json.load(open(AUDIT_MAP, encoding="utf-8"))
    models = audit_map["models17"]
    counts = {m: dict.fromkeys(jv.CELLS, 0) for m in models}
    for r in json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))["rows"]:
        if r["model"] in counts:
            counts[r["model"]][f"{r['sb_string']}{r['em']}"] += 1
    shares = {m: {c: v[c] / sum(v.values()) for c in jv.CELLS} | {"N": sum(v.values())} for m, v in counts.items()}
    return jv, models, shares, audit_map


def test_new_census_reproduces_old_shares_and_817(anchor_inputs):
    jv, models, shares, audit_map = anchor_inputs
    old_shares, _, _ = jv.census()
    assert audit_map["disagreements_17_sb_string_vs_em"] == 817
    for m in models:
        assert shares[m] == pytest.approx(old_shares[m], abs=0)


@pytest.mark.parametrize("indet", ["as_incorrect", "drop"])
def test_audit_cells_reproduce_142_32_1_145(anchor_inputs, indet):
    _, _, _, audit_map = anchor_inputs
    cells = audit_map["audit_cells"][f"17|k2_string|string|{indet}"]
    old = json.load(open(ROOT / "results" / "flipbudget" / "joint_verifier_results.json"))[indet]["audit_cells"]
    assert cells == old
    if indet == "as_incorrect":
        assert {c: v["n"] for c, v in cells.items()} == {"11": 142, "10": 32, "01": 1, "00": 145}


@pytest.mark.parametrize("indet", ["as_incorrect", "drop"])
def test_certify_module_reproduces_anchor_at_eps0(anchor_inputs, indet):
    """G2.4: the new implementation (eps=0, CP, legacy unpaired sampling) gives 84 / 101."""
    import math

    from flipbudget import certify

    jv, models, shares, audit_map = anchor_inputs
    cells = audit_map["audit_cells"][f"17|k2_string|string|{indet}"]
    conc = {c: certify.cp_interval(cells[c]["x_correct"], cells[c]["n"], certify.sidak(2)) for c in ("11", "00")}
    unaudited = {c: (0.0, 1.0) for c in ("10", "01")}

    def legacy_h(j, k, eps):
        a = [shares[m]["11"] + shares[m]["10"] for m in (j, k)]
        return jv.Z_SAMP * math.sqrt(sum(x * (1 - x) / shares[m]["N"] for x, m in zip(a, (j, k))))

    r = certify.evaluate(models, shares, lambda j, k: (conc, unaudited, unaudited), legacy_h, 0.0)
    assert (r["combined_resolved"], r["audit_resolved"]) == (84, 101)


@pytest.mark.parametrize("indet", ["as_incorrect", "drop"])
def test_anchor_84_101_118(anchor_inputs, indet):
    jv, models, shares, audit_map = anchor_inputs
    cells = audit_map["audit_cells"][f"17|k2_string|string|{indet}"]
    b11 = jv.cp(cells["11"]["x_correct"], cells["11"]["n"], jv.lvl(2))
    b00 = jv.cp(cells["00"]["x_correct"], cells["00"]["n"], jv.lvl(2))
    r = jv.evaluate(shares, models, ("shared", b11, b00), lambda m, c: (0.0, 1.0))
    assert (r["combined_resolved"], r["audit_resolved"], jv.sampling_ceiling(shares, models)) == (84, 101, 118)
