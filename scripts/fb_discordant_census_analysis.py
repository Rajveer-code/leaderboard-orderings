"""Census labels -> per-domain bounds for the discordant cells (TC_EXPANSION_PREREGISTRATION
Amendment 6). Runs on whatever census batches exist; a domain is complete only when every one
of its responses carries an admitted label.

Labels are pooled by response across the three audits: the census sheet
(labeling/exports/census/), the registered audit's discordant rows (labeling/exports/, primary
labeller), and the existing 400-item audit. Each response contributes its reader-derived verdict
under the harness routine (primary) and Math-Verify (sensitivity). Indeterminate answers are
scored incorrect (primary) or treated as unlabelled (sensitivity). Batches failing the gate are
excluded; the D8 variant restores those whose only failure is the extraction check.

For each discordant (model, cell) domain the output records x (credited among labelled), n
(labelled) and size (all responses in the domain), which scripts/fb_certify.py turns into
census bounds: exact when n == size, the logical range otherwise.

Run: python scripts/fb_discordant_census_analysis.py
Out: results/flipbudget/discordant_census_labels.json
"""
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget.verifiers import human_correct, is_indeterminate  # noqa: E402
from fb_audit_v3_analysis import EXPORTS, load_batches, read_sheet  # noqa: E402

CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
SHEET = ROOT / "labeling" / "discordant_census_l1.csv"
PROV = ROOT / "results" / "analysis" / "discordant_census_provenance.json"
V3_SHEET = ROOT / "labeling" / "audit_v3_l1.csv"
V3_PROV = ROOT / "results" / "analysis" / "audit_v3_provenance.json"
EXISTING = ROOT / "results" / "flipbudget" / "audit_k3_map.json"
OUT = Path(os.environ.get("FB_CENSUS_OUT", ROOT / "results" / "flipbudget" / "discordant_census_labels.json"))
FIELDS = ("em", "sb_sympy", "mv")
REFS = ("sympy", "mv")


def verdicts(records, prov, gold, solution):
    """{(model, item): {"indet": bool, ref: 0/1/None}} for records whose uid is in prov."""
    out = {}
    for uid, r in records.items():
        p = prov.get(uid)
        if p is None:
            continue
        a = r["your_answer"]
        indet = is_indeterminate(a)
        out[(p["model"], p["item_id"])] = {"indet": indet, **{
            ref: None if indet else int(human_correct(a, solution[p["item_id"]] if ref == "mv" else gold[p["item_id"]],
                                                      ref)) for ref in REFS}}
    return out


def main():
    if not any((EXPORTS / "census").glob("*.csv")):
        print(f"no census batches in {EXPORTS / 'census'} yet; nothing written, so the certification "
              "is unchanged")
        return
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    cell = {(r["model"], r["item_id"]): "".join(str(r[f]) for f in FIELDS) for r in census["rows"]}
    size = Counter((m, c) for (m, _), c in cell.items() if c not in ("000", "111"))
    from fb_census_k3 import load_rows
    ref_rows = load_rows(census["models"][:1])[0]  # gold and solution are per item, not per model
    gold = {r["item_id"]: r["gold"] for r in ref_rows}
    solution = {r["item_id"]: r["solution"] for r in ref_rows}

    census_prov = {r["uid"]: r for r in json.load(open(PROV, encoding="utf-8"))["rows"]}
    v3_prov = {r["uid"]: r for r in json.load(open(V3_PROV, encoding="utf-8"))["rows"]
               if r["cell"] not in ("000", "111")}
    existing = {(it["model"], it["item_id"]): {"indet": it["indeterminate"],
                                               **{ref: None if it["indeterminate"] else int(it["ystar"][ref])
                                                  for ref in REFS}}
                for it in json.load(open(EXISTING, encoding="utf-8"))["items"]
                if it["k3_sympy"] not in ("000", "111")}

    out = {"bounds_input": {}, "completeness": {}, "excluded_batches": {}}
    for restore, tag in ((False, "gate"), (True, "d8_extraction_restored")):
        c_rec, c_exc = load_batches(read_sheet(SHEET), subdir="census", restore_extraction_only=restore)
        v_rec, v_exc = load_batches(read_sheet(V3_SHEET), restore_extraction_only=restore)
        labels = dict(existing)
        labels.update(verdicts(v_rec, v3_prov, gold, solution))
        labels.update(verdicts(c_rec, census_prov, gold, solution))
        out["excluded_batches"][tag] = {"census": c_exc, "registered": v_exc}
        for ref in REFS:
            for indet_mode in ("as_incorrect", "dropped"):
                variant = f"{tag}|{ref}|{indet_mode}"
                agg = {}
                for (m, item), v in labels.items():
                    c = cell.get((m, item))
                    if c is None or c in ("000", "111"):
                        continue
                    if v["indet"] and indet_mode == "dropped":
                        continue  # treated as unlabelled: the domain's bounds widen
                    x, n = agg.get((m, c), (0, 0))
                    agg[(m, c)] = (x + (0 if v["indet"] else v[ref]), n + 1)
                out["bounds_input"][variant] = {f"{m}|{c}": {"x": agg.get((m, c), (0, 0))[0],
                                                            "n": agg.get((m, c), (0, 0))[1], "size": size[(m, c)]}
                                                for (m, c) in size}
                done = sum(agg.get(d, (0, 0))[1] == size[d] for d in size)
                out["completeness"][variant] = {"domains": len(size), "complete": done,
                                                "labelled": sum(v[1] for v in agg.values()),
                                                "responses": sum(size.values())}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    primary = out["completeness"]["gate|sympy|as_incorrect"]
    print(f"wrote {OUT}: {primary['labelled']} of {primary['responses']} discordant responses labelled; "
          f"{primary['complete']} of {primary['domains']} domains complete")


if __name__ == "__main__":
    main()
