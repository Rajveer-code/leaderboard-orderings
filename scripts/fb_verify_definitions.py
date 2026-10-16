"""Re-implement four definitions exactly as the paper words them and check that each reproduces
the number the paper reports. Deliberately independent of the production code paths.

  reversal    Appendix G: reversed pairs over pairs strictly ordered by both scores (ties excluded)
  compliance  Appendix I: append "I hope it is correct.", re.search the 2024 template pattern
  taxonomy    Appendix I: runtime -> extraction (template / box / parse) -> judgement, in order
  sixteen     Appendix F: xFinder's verdict appended; 0000/1111 concordant; occupied cells, cells
              left by the move, and the audited concordant labels

Run: python scripts/fb_verify_definitions.py   (needs the local response cache for compliance)
Out: results/analysis/definition_checks.json; exit status 1 on any mismatch
"""
import gzip
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results" / "flipbudget"
OUT = ROOT / "results" / "analysis" / "definition_checks.json"
TEMPLATE = re.compile(r"Final Answer: The final answer is(.*?). I hope it is correct.")


def compliant(response: str) -> bool:
    """Appendix I, word for word: append the closing sentence, then search the pattern."""
    return TEMPLATE.search(response + "I hope it is correct.") is not None


def truthy(v) -> bool:
    return v in (True, "True", 1, "1")


def taxonomy_class(r, is_compliant):
    """Appendix I's decision tree, applied in order."""
    if any(truthy(r.get(f)) for f in ("em_rec_timeout", "em_rec_killed", "sb_timeout", "sb_killed",
                                       "mv_timeout", "mv_killed")):
        return "runtime"
    subs = []
    if int(r["em"]) == 0 and not is_compliant:
        subs.append("template")
    if int(r["sb_sympy"]) == 0 and not truthy(r["sb_found"]):
        subs.append("box")
    if int(r["mv"]) == 0 and not truthy(r["mv_pred_parsed"]):
        subs.append("parse")
    return "extraction:" + "+".join(subs) if subs else "judgement"


def main():
    checks = {}
    census = json.load(gzip.open(RES / "census_k3.json.gz", "rt", encoding="utf-8"))
    rows = census["rows"]

    # reversal statistic, from the stored pair counts against a direct recount on the roster
    shift = json.load(open(RES / "board_verifier_shift.json", encoding="utf-8"))
    pc = shift["board"]["pair_counts"]
    checks["reversal_board"] = {"reported": shift["board"]["discordant_pair_share"],
                                "from_definition": pc["reversed"] / pc["strictly_ordered_by_both"]}
    rs = shift["roster_shift"]  # the roster's before/after difference; recount needs both scores
    checks["reversal_roster_pairs"] = {"reported": shift["roster"]["pair_counts"]["pairs"],
                                       "from_definition": len(list(itertools.combinations(rs, 2)))}

    # template compliance, from the responses
    from fb_census_k3 import load_rows
    text, _ = load_rows(census["models"])
    resp = {(t["model"], t["item_id"]): t["response"] for t in text}
    comp = {(r["model"], r["item_id"]): isinstance(resp.get((r["model"], r["item_id"])), str)
            and compliant(resp[(r["model"], r["item_id"])]) for r in rows}
    tax = json.load(open(RES / "extraction_taxonomy.json", encoding="utf-8"))
    checks["template_compliant"] = {"reported": tax["board"]["template_compliant"],
                                    "from_definition": sum(comp.values())}

    # the taxonomy
    disc = [r for r in rows if len({r["em"], r["sb_sympy"], r["mv"]}) > 1]
    cls = Counter(taxonomy_class(r, comp[(r["model"], r["item_id"])]) for r in disc)
    top = Counter(c.split(":")[0] for c in cls.elements())
    checks["taxonomy_top"] = {"reported": tax["taxonomy"]["top"], "from_definition": dict(top)}
    checks["taxonomy_extraction_sub"] = {"reported": tax["taxonomy"]["extraction_sub"],
                                         "from_definition": {c: n for c, n in cls.items() if c.startswith("extraction")}}

    # the sixteen cells
    xf = json.load(gzip.open(RES / "xfinder_verdicts.json.gz", "rt", encoding="utf-8"))
    v = xf["verdicts"][xf["primary_arm"]]
    cell4 = Counter("".join(str(int(r[f])) for f in ("em", "sb_sympy", "mv")) + str(v[f"{r['model']}|{r['item_id']}"]["verdict"])
                    for r in rows)
    cert = json.load(open(RES / "certify_results.json", encoding="utf-8"))["K4"]["train"]
    checks["kfour_occupied"] = {"reported": sum(1 for n in cert["cells"].values() if n),
                                "from_definition": sum(1 for n in cell4.values() if n)}
    checks["kfour_discordant"] = {"reported": cert["n_discordant"],
                                  "from_definition": sum(n for c, n in cell4.items() if c not in ("0000", "1111"))}
    audit = json.load(open(RES / "audit_k3_map.json", encoding="utf-8"))["items"]
    conc4 = Counter(it["k3_sympy"] + str(v[f"{it['model']}|{it['item_id']}"]["verdict"]) for it in audit
                    if it.get("k3_sympy") in ("000", "111"))
    checks["kfour_audit_concordant"] = {"reported": [cert["inputs"]["audit_pooled"]["1111"]["n"],
                                                     cert["inputs"]["audit_pooled"]["0000"]["n"]],
                                        "from_definition": [conc4["1111"], conc4["0000"]]}

    bad = {k: c for k, c in checks.items() if c["reported"] != c["from_definition"]
           and not (isinstance(c["reported"], float) and abs(c["reported"] - c["from_definition"]) < 1e-12)}
    json.dump({"checks": checks, "mismatches": sorted(bad)}, open(OUT, "w", encoding="utf-8"), indent=1)
    for k, c in checks.items():
        print(f"{'OK ' if k not in bad else 'BAD'} {k}: reported {c['reported']} | from definition {c['from_definition']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
