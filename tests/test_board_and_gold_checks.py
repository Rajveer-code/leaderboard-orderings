"""The board response-date sample and the gold-answer consistency check are internally consistent."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BD = json.load(open(ROOT / "results" / "flipbudget" / "board_response_dates.json", encoding="utf-8"))
GC = json.load(open(ROOT / "results" / "analysis" / "gold_consistency.json", encoding="utf-8"))
SHIFT = json.load(open(ROOT / "results" / "flipbudget" / "board_verifier_shift.json", encoding="utf-8"))


def test_board_sample_counts():
    assert BD["n_board"] == SHIFT["n_models_both"]
    assert len(BD["per_model"]) == BD["n_sampled"]
    checked = [v for v in BD["per_model"].values() if v["status"] == "checked"]
    assert len(checked) == BD["n_checked"] == BD["n_sampled"] - BD["n_unavailable"]
    for v in checked:  # the pass flag agrees with the recorded dates
        assert v["all_on_or_before_snapshot"] == (v["last"] <= BD["snapshot_date"])
    assert BD["n_unchanged"] == sum(v["all_on_or_before_snapshot"] for v in checked)


def test_gold_counts():
    assert GC["n_items"] == 1324
    assert GC["n_agree"] + GC["n_disagree"] == GC["n_items"]
    assert len(GC["disagreements"]) == GC["n_disagree"]


def test_ft_check_counts():
    gf = json.load(open(ROOT / "results" / "analysis" / "gold_ft.json", encoding="utf-8"))
    assert gf["n_items"] == 1324 and len(gf["affected"]) == gf["n_affected"]
    assert len(gf["labels"]) == gf["n_audited_labels_on_affected"]
    assert gf["n_in_concordant_cells"] == sum(x["cell"] in ("000", "111") for x in gf["labels"])
    assert gf["n_verdicts_changed"] == sum(x["ystar_used"] != x["ystar_without_ft"] for x in gf["labels"])
    for v in gf["affected"].values():  # the rule must actually remove letters from a LaTeX command
        assert "ft" in v["boxed"] and v["stored"] != v["boxed"]


def test_row_provenance_counts():
    rp = json.load(open(ROOT / "results" / "flipbudget" / "board_row_provenance.json", encoding="utf-8"))
    assert sum(rp["classes"].values()) == rp["n_models"] == SHIFT["n_models_both"]
    assert rp["reversal"]["all"]["reversed_pairs"] == SHIFT["board"]["pair_counts"]["reversed"]
    srd = rp["same_run_detail"]
    assert srd["board"]["n"] == rp["same_run_pairs"] == rp["reversal"]["same_run"]["models"]
    pc = srd["board"]["pair_counts"]
    assert abs(pc["reversed"] / pc["strictly_ordered_by_both"] - srd["board"]["discordant_pair_share"]) < 1e-12
    assert rp["snapshot_score_equals_same_run_before"] <= rp["same_run_pairs"]
    for m, v in rp["per_model"].items():  # an attributable model's displayed score is a re-graded value
        if v["class"] == "attributable":
            assert any(abs(v["displayed"] - x) <= 1e-9 for x in v["regraded_values"])
