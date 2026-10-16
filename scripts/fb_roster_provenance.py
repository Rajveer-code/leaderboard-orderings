"""How the 27-model roster came to exist -- snapshot of the selection record (MASTER_PLAN T11).

The roster was not drawn for this paper. It descends from a selection made for an earlier
study in a companion repository: every Open LLM Leaderboard detail dataset was catalogued,
ranked by download count, capped at four per organisation, and the top sixty kept. A download
sweep over those sixty was interrupted; the completed downloads were scored; the existing
400-item human audit then drew its items from them; and this paper's roster is the set of
models that received at least one audited item.

This script reads the companion repository's own records (read-only), recomputes each step's
count, and writes them with the sha256 of every source file, so the paper can state the
selection with generated numbers and a reader can check them.

Run: python scripts/fb_roster_provenance.py   Out: results/flipbudget/roster_provenance.json
"""
import hashlib
import json
import os
from pathlib import Path

# PATHS
ROOT = Path(__file__).resolve().parents[1]
KS = Path(os.environ.get("FB_COMPANION_REPO", ROOT.parent / "knowledgeshift"))
CATALOGUE = KS / "results" / "analysis" / "leaderboard_details_catalogue.json"
SELECTION = KS / "results" / "analysis" / "l1_selection.json"
SCORED = KS / "results" / "analysis" / "l1_leaderboard_rescore.json"
RULE_DOC = KS / "docs" / "L1_ACCESS_LIST.md"
ROSTER = ROOT / "results" / "flipbudget" / "e4_mathhard_per_model.json"
OUT = ROOT / "results" / "flipbudget" / "roster_provenance.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def model_of(dataset_id: str) -> str:
    return dataset_id.split("/", 1)[1].removesuffix("-details").replace("__", "/", 1)


def main():
    catalogue = json.load(open(CATALOGUE, encoding="utf-8"))
    selection = json.load(open(SELECTION, encoding="utf-8"))
    scored = [r["model"] for r in json.load(open(SCORED, encoding="utf-8"))["results"]]
    roster = sorted(json.load(open(ROSTER, encoding="utf-8"))["per_model"])
    order = [model_of(r["id"]) for r in selection]

    downloads = [r["downloads"] for r in selection]
    orgs = [m.split("/")[0] for m in order]
    per_org = {o: orgs.count(o) for o in set(orgs)}
    positions = sorted(order.index(m) + 1 for m in roster)  # 1-based rank in download order

    assert downloads == sorted(downloads, reverse=True), "selection is not in download order"
    assert set(roster) <= set(scored) <= set(order), "roster is not nested in scored in selected"

    rule_text = RULE_DOC.read_text(encoding="utf-8")
    rule = rule_text[rule_text.index("## Selection rule"):rule_text.index("**60 datasets")].strip()

    out = {
        "n_detail_datasets_catalogued": len(catalogue),
        "n_selected": len(selection),
        "max_per_organisation": max(per_org.values()),
        "n_organisations_selected": len(per_org),
        "selection_downloads_range": [min(downloads), max(downloads)],
        "n_scored_after_interrupted_sweep": len(scored),
        "n_roster": len(roster),
        "scored_not_in_roster": sorted(set(scored) - set(roster)),
        "roster_rank_in_download_order": positions,
        "roster_deepest_rank": max(positions),
        "selection_rule_as_registered": rule,
        "sources": {str(p.relative_to(KS)).replace("\\", "/"): sha256(p)
                    for p in (CATALOGUE, SELECTION, SCORED, RULE_DOC)},
        "companion_repository_note": "read-only; named by relative path so the record does not "
                                     "depend on where the repository sits on disk",
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}")
    for k in ("n_detail_datasets_catalogued", "n_selected", "max_per_organisation",
              "n_organisations_selected", "n_scored_after_interrupted_sweep", "n_roster",
              "roster_deepest_rank", "scored_not_in_roster"):
        print(f"  {k}: {out[k]}")


if __name__ == "__main__":
    main()
