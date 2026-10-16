"""G1.5 -- map the existing 400-item human audit onto K=2 and K=3 verifier cells.

Human reference Y* under three equivalence routines (verifiers.human_correct): string
(historical), sympy (faithful harness), mv (Math-Verify). Indeterminate answers: Y*=0
(primary) or dropped (sensitivity). Cells use the census verdicts:
  K=2 historical: (sb_string, em)        -- must reproduce 142/32/1/145 on the 17 models
  K=2 faithful:   (sb_sympy, em)
  K=3:            (em, sb_sympy, mv) and (em, sb_string, mv)

Run after fb_census_k3.py:  python scripts/fb_audit_k3_map.py
Out: results/flipbudget/audit_k3_map.json
"""
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from flipbudget.verifiers import human_correct, is_indeterminate  # noqa: E402

CENSUS = ROOT / "results" / "flipbudget" / "census_k3.json.gz"
KEY = ROOT / "results" / "analysis" / "mathhard_labelling_key.json"
ANSWERS = ROOT / "results" / "analysis" / "mathhard_l1.csv"
POP17 = ROOT / "results" / "analysis" / "tc_expansion_population.json"
OUT = ROOT / "results" / "flipbudget" / "audit_k3_map.json"
CELLINGS = {"k2_string": ("sb_string", "em"), "k2_sympy": ("sb_sympy", "em"),
            "k3_sympy": ("em", "sb_sympy", "mv"), "k3_string": ("em", "sb_string", "mv")}
REFS = ("string", "sympy", "mv")


def cell(row, fields):
    return "".join(str(row[f]) for f in fields)


def main():
    census = json.load(gzip.open(CENSUS, "rt", encoding="utf-8"))
    rows = {(r["model"], r["item_id"]): r for r in census["rows"]}
    models17 = sorted({r["model"] for r in json.load(open(POP17, encoding="utf-8"))["population"]})
    key = json.load(open(KEY, encoding="utf-8"))
    with open(ANSWERS, newline="", encoding="utf-8") as f:
        answers = {r["uid"]: r["your_answer"] for r in csv.DictReader(f)}

    from fb_census_k3 import load_rows  # every model has all 1,324 items; one model gives every solution
    solution = {r["item_id"]: r["solution"] for r in load_rows(census["models"][:1])[0]}
    out = {"n_key": len(key), "models17": models17}
    r17 = [r for r in census["rows"] if r["model"] in models17]
    out["disagreements_17_sb_string_vs_em"] = sum(r["sb_string"] != r["em"] for r in r17)
    out["disagreements_17_sb_sympy_vs_em"] = sum(r["sb_sympy"] != r["em"] for r in r17)
    out["census_cells"] = {name: {scope: dict(Counter(cell(r, f) for r in rs))
                                  for scope, rs in (("17", r17), ("27", census["rows"]))}
                           for name, f in CELLINGS.items()}

    items, unmatched = [], []
    for k in key:
        r = rows.get((k["model"], k["item_id"]))
        if r is None:
            unmatched.append(k["uid"])
            continue
        a = answers.get(k["uid"])
        indet = is_indeterminate(a)
        # key["gold"] is already normalize_final_answer'd (which deletes "ft": \infty -> \iny);
        # harmless for string/sympy (they re-normalize); Math-Verify parses gold from the solution.
        ystar = {ref: None if indet else human_correct(a, solution[k["item_id"]] if ref == "mv" else k["gold"], ref)
                 for ref in REFS}
        items.append({"uid": k["uid"], "model": k["model"], "item_id": k["item_id"], "in17": k["model"] in models17,
                      "key_stratum": k["stratum"], "indeterminate": indet, "ystar": ystar,
                      **{name: cell(r, f) for name, f in CELLINGS.items()}})
    out["unmatched_key_uids"] = unmatched
    out["ystar_disagreements_between_refs"] = {
        f"{a}_vs_{b}": sum(i["ystar"][a] != i["ystar"][b] for i in items if not i["indeterminate"])
        for a, b in (("string", "sympy"), ("string", "mv"), ("sympy", "mv"))}

    def counts(scope, cells_name, ref, indet):
        c = defaultdict(lambda: [0, 0])
        for i in items:
            if scope == "17" and not i["in17"]:
                continue
            if i["indeterminate"] and indet == "drop":
                continue
            y = 0 if i["indeterminate"] else i["ystar"][ref]
            c[i[cells_name]][0] += y
            c[i[cells_name]][1] += 1
        return {k: {"x_correct": v[0], "n": v[1]} for k, v in sorted(c.items())}

    out["audit_cells"] = {f"{scope}|{name}|{ref}|{indet}": counts(scope, name, ref, indet)
                          for scope in ("17", "27") for name in CELLINGS for ref in REFS
                          for indet in ("as_incorrect", "drop")}
    out["items"] = items
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("17-model disagreements sb_string vs em:", out["disagreements_17_sb_string_vs_em"],
          "| sb_sympy vs em:", out["disagreements_17_sb_sympy_vs_em"])
    print("unmatched key rows:", len(unmatched), "| Y* ref disagreements:", out["ystar_disagreements_between_refs"])
    for k in ("17|k2_string|string|as_incorrect", "17|k2_sympy|sympy|as_incorrect", "27|k3_sympy|sympy|as_incorrect"):
        print(k, out["audit_cells"][k])


if __name__ == "__main__":
    main()
